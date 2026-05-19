"""
trt_pose_engine.py — Ultralytics 완전 우회 + Zero-Copy 최적화

최적화:
  1. GPU output 파싱: 7200개 → 18개만 CPU 복사 (-2ms)
  2. torch GPU 전처리: resize + BGR→RGB + normalize 전부 GPU (-3ms)
  3. Depth smoothing: patch sampling + temporal smoothing (안정성)
  4. 버퍼 사전 할당: 매 프레임 malloc 없음
  5. ONNX→엔진 자동 빌드 (버전 불일치 시)

출력 shape: (1, 300, 24) = xyxy(4) + conf(1) + cls(1) + 6kpt*3(18)

사용법:
    engine = TRTPoseEngine('yolo26s-lower6-v2-640.engine', imgsz=640)
    engine.load()
    result = engine.predict(bgr_image)
    coords_3d = engine.get_3d_coords(result, depth_map, patch_size=5, alpha=0.7)
"""

import os
import re
import numpy as np
import cv2

try:
    import tensorrt as trt
    import torch
    HAS_TRT = True
except ImportError:
    HAS_TRT = False


class TRTPoseEngine:
    KEYPOINT_MAP = {
        0: "left_hip",    1: "right_hip",
        2: "left_knee",   3: "right_knee",
        4: "left_ankle",  5: "right_ankle",
    }

    def __init__(self, engine_path, imgsz=640, conf=0.25, iou=0.7, num_kpts=6,
                 use_cpp_trt=False, use_cuda_preprocess=False):
        """TRT pose inference engine.

        Args:
            use_cpp_trt: if True, use hwalker_trt_runner C++ extension instead
                of Python TRT context.execute_async_v3(). Drop-in replacement;
                preprocess + postprocess paths identical. (Sprint 1 Phase 2 Week 2 —
                ABORTED: premise wrong, no gain, cosine regression. Kept opt-in for
                regression testing; do not enable in production.)
            use_cuda_preprocess: if True, use hwalker_cuda_preprocess C++ extension
                (single fused CUDA kernel) instead of torch op chain in _preprocess_gpu().
                Expected ~1.0-1.3 ms gain (Sprint 1 Phase 2 Week 3). Drop-in replacement:
                identical scale/pad_w/pad_h return values. Caller should build cpp/
                first (scripts/build_cpp.sh).
        """
        self.engine_path = engine_path
        self.imgsz = imgsz
        self.conf_thresh = conf
        self.iou_thresh = iou
        self.num_kpts = num_kpts
        self.is_loaded = False
        self.use_cpp_trt = bool(use_cpp_trt)
        self.use_cuda_preprocess = bool(use_cuda_preprocess)

        # Python TRT state (only populated when use_cpp_trt=False)
        self._context = None
        self._engine = None
        # C++ TRT state (only populated when use_cpp_trt=True)
        self._cpp_runner = None
        self._input_name = None
        self._output_name = None

        # CUDA preprocess state (only populated when use_cuda_preprocess=True)
        self._cuda_preprocessor = None

        # Shared GPU I/O buffers (both paths use these)
        self._stream = None
        self._input_tensor = None
        self._output_tensor = None
        self._pad_tensor = None

        # letterbox 캐시
        self._cached_src_shape = None
        self._cached_scale = None
        self._cached_pad = None
        self._cached_new_size = None

        # depth temporal smoothing 상태
        self._prev_3d = {}  # {joint_name: np.array([x,y,z])}

    # ── 엔진 빌드 ────────────────────────────────────────────────────────────

    @staticmethod
    def build_engine_from_onnx(onnx_path, engine_path, imgsz, fp16=True):
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  [TRT Build] ONNX error: {parser.get_error(i)}")
                raise RuntimeError("ONNX parse failed")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        shape = (1, 3, imgsz, imgsz)
        profile.set_shape(input_name, shape, shape, shape)
        config.add_optimization_profile(profile)
        print(f"  [TRT Build] Building (imgsz={imgsz}, FP16)... 1-2분")
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Engine build failed")
        with open(engine_path, 'wb') as f:
            f.write(engine)
        print(f"  [TRT Build] Saved: {engine_path}")

    # ── 로드 ─────────────────────────────────────────────────────────────────

    def load(self):
        if not HAS_TRT:
            raise ImportError("tensorrt or torch not installed")

        logger = trt.Logger(trt.Logger.WARNING)
        engine_loaded = False

        # .direct.engine → .engine → ONNX 빌드 순서
        direct_path = self.engine_path.replace('.engine', '.direct.engine')
        for try_path in [direct_path, self.engine_path]:
            if not os.path.exists(try_path):
                continue
            try:
                with open(try_path, 'rb') as f:
                    self._engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
                if self._engine is not None:
                    engine_loaded = True
                    self.engine_path = try_path
                    break
            except Exception:
                pass

        if not engine_loaded:
            onnx_path = self.engine_path.replace('.engine', '.onnx')
            if not os.path.exists(onnx_path):
                base = re.sub(r'-\d+\.engine$', '.onnx', self.engine_path)
                if os.path.exists(base):
                    onnx_path = base
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"No engine or ONNX: {self.engine_path}")
            direct_engine = self.engine_path.replace('.engine', '.direct.engine')
            self.build_engine_from_onnx(onnx_path, direct_engine, self.imgsz)
            self.engine_path = direct_engine
            with open(self.engine_path, 'rb') as f:
                self._engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        self._input_name = self._engine.get_tensor_name(0)
        self._output_name = self._engine.get_tensor_name(1)
        input_shape = list(self._engine.get_tensor_shape(self._input_name))
        output_shape = list(self._engine.get_tensor_shape(self._output_name))
        print(f"  [TRT] input: {self._input_name} {input_shape}")
        print(f"  [TRT] output: {self._output_name} {output_shape}")

        self._stream = torch.cuda.Stream()

        # GPU 버퍼 사전 할당 (shared between Python + C++ paths)
        self._input_tensor = torch.zeros(input_shape, dtype=torch.float32, device='cuda')
        self._output_tensor = torch.empty(output_shape, dtype=torch.float32, device='cuda')
        self._pad_tensor = torch.full(
            (1, 3, self.imgsz, self.imgsz), 114.0 / 255.0,
            dtype=torch.float32, device='cuda'
        )

        if self.use_cpp_trt:
            # C++ TRT runner (Sprint 1 Phase 2 Week 2 — ABORTED, kept opt-in)
            self._context = None
            try:
                import hwalker_trt_runner  # type: ignore
            except ImportError:
                try:
                    from perception.realtime import hwalker_trt_runner  # type: ignore
                except ImportError as e:
                    raise RuntimeError(
                        "use_cpp_trt=True but hwalker_trt_runner not found. "
                        "Build first: scripts/build_cpp.sh"
                    ) from e
            self._cpp_runner = hwalker_trt_runner.TrtRunner(self.engine_path)
            print(f"  [TRT] using C++ TrtRunner (hwalker_trt_runner)")
        else:
            # Python TRT path (original)
            self._context = self._engine.create_execution_context()
            self._context.set_tensor_address(
                self._input_name, self._input_tensor.data_ptr()
            )
            self._context.set_tensor_address(
                self._output_name, self._output_tensor.data_ptr()
            )
            print(f"  [TRT] using Python TRT context")

        # CUDA preprocess kernel (Sprint 1 Phase 2 Week 3)
        if self.use_cuda_preprocess:
            try:
                import hwalker_cuda_preprocess  # type: ignore
            except ImportError:
                try:
                    from perception.realtime import hwalker_cuda_preprocess  # type: ignore
                except ImportError as e:
                    raise RuntimeError(
                        "use_cuda_preprocess=True but hwalker_cuda_preprocess not found. "
                        "Build first: scripts/build_cpp.sh (requires CUDA Toolkit)"
                    ) from e
            self._cuda_preprocessor = hwalker_cuda_preprocess.CudaPreprocessor(
                self.imgsz, max_h=1200, max_w=1920, max_channels=4
            )
            print(f"  [Preprocess] using CUDA kernel (hwalker_cuda_preprocess)")

        # 워밍업 (all paths)
        dummy = np.zeros((600, 960, 3), dtype=np.uint8)
        for _ in range(3):
            self.predict(dummy)

        self.is_loaded = True
        bits = ["torch GPU preprocess" if not self.use_cuda_preprocess else "CUDA kernel preprocess",
                "GPU output parsing"]
        if self.use_cpp_trt:
            bits.append("C++ infer")
        print(f"  [TRT] Ready ({', '.join(bits)})")

    # ── 전처리 (torch GPU) ───────────────────────────────────────────────────

    def _calc_letterbox(self, h, w):
        if self._cached_src_shape == (h, w):
            return self._cached_scale, self._cached_pad, self._cached_new_size
        sz = self.imgsz
        scale = min(sz / h, sz / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h, pad_w = (sz - new_h) // 2, (sz - new_w) // 2
        self._cached_src_shape = (h, w)
        self._cached_scale = scale
        self._cached_pad = (pad_w, pad_h)
        self._cached_new_size = (new_w, new_h)
        return scale, (pad_w, pad_h), (new_w, new_h)

    def _preprocess_gpu(self, image, is_bgra=False):
        """torch GPU: upload → channel swap → resize → normalize → letterbox.

        If use_cuda_preprocess=True, replaces the whole chain with a single
        fused CUDA kernel (hwalker_cuda_preprocess). Returns identical scale,
        pad_w, pad_h. Kernel runs on self._stream (same stream as TRT) to
        avoid cross-stream synchronization with the inference that follows.
        """
        h, w = image.shape[:2]
        scale, (pad_w, pad_h), (new_w, new_h) = self._calc_letterbox(h, w)

        # ── CUDA kernel fast path (Sprint 1 Phase 2 Week 3) ────────────────
        if self._cuda_preprocessor is not None:
            # Stream choice (debug knob — defaults to TRT stream, current behavior):
            #   HWALKER_PREPROC_STREAM=trt     (default) — kernel + TRT on same stream
            #   HWALKER_PREPROC_STREAM=null    — legacy NULL stream (auto-syncs all)
            #   HWALKER_PREPROC_STREAM=default — torch's current stream (per-thread default)
            mode = os.environ.get("HWALKER_PREPROC_STREAM", "trt").lower()
            if mode == "null":
                stream_h = 0   # legacy default stream: implicit sync with all
            elif mode == "default":
                stream_h = torch.cuda.current_stream().cuda_stream
            else:
                stream_h = self._stream.cuda_stream
            ok = self._cuda_preprocessor.process(
                image,
                self._input_tensor.data_ptr(),
                bool(is_bgra),
                stream_h,
            )
            if not ok:
                raise RuntimeError("hwalker_cuda_preprocess.process() returned False")
            # If preprocess ran on a different stream than self._stream, ensure
            # TRT (which runs on self._stream) waits for preprocess to finish.
            # NULL stream syncs automatically. Per-thread default does NOT.
            if mode == "default":
                self._stream.wait_stream(torch.cuda.current_stream())
            return scale, pad_w, pad_h

        # ── torch reference path (original) ────────────────────────────────
        # numpy → GPU (한 번만)
        t = torch.from_numpy(np.ascontiguousarray(image)).cuda(non_blocking=True)

        # 채널 swap (GPU 인덱싱)
        if is_bgra:
            t = t[:, :, [2, 1, 0]]   # BGRA → RGB
        else:
            t = t[:, :, [2, 1, 0]]   # BGR → RGB

        # HWC→CHW + float + normalize (GPU)
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

        # GPU resize
        t = torch.nn.functional.interpolate(
            t, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # letterbox 패딩 (사전 할당 버퍼)
        self._input_tensor.copy_(self._pad_tensor)
        self._input_tensor[0, :, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = t[0]

        return scale, pad_w, pad_h

    # ── 추론 ─────────────────────────────────────────────────────────────────

    def predict(self, image, is_bgra=False, _profile=None):
        """Run inference.

        Args:
            image: HxWxC numpy (BGR or BGRA per is_bgra).
            is_bgra: input is BGRA (skips conversion).
            _profile: optional dict — populated with per-stage ms:
                'preprocess_ms', 'infer_ms', 'postprocess_ms'.
                ⚠ Enables torch.cuda.synchronize() between stages — may
                  add jitter. Use ONLY for diagnostic, not production hot path.
        """
        import time
        h, w = image.shape[:2]

        if _profile is not None:
            t0 = time.perf_counter()

        scale, pad_w, pad_h = self._preprocess_gpu(image, is_bgra=is_bgra)

        if _profile is not None:
            torch.cuda.synchronize()   # wait preprocess GPU ops
            t1 = time.perf_counter()
            _profile['preprocess_ms'] = (t1 - t0) * 1000.0

        # TRT 실행 — Python TRT context 또는 C++ TrtRunner (use_cpp_trt 분기)
        with torch.cuda.stream(self._stream):
            if self._cpp_runner is not None:
                # C++ path: pass pre-allocated tensor data_ptr + stream handle
                # Returns False on TRT internal failure; raise to surface upstream.
                ok = self._cpp_runner.infer(
                    self._input_tensor.data_ptr(),
                    self._output_tensor.data_ptr(),
                    self._stream.cuda_stream,
                )
                if not ok:
                    raise RuntimeError("hwalker_trt_runner.infer() returned False")
            else:
                # Python path: pre-bound buffers via set_tensor_address (in load())
                self._context.execute_async_v3(self._stream.cuda_stream)
            self._stream.synchronize()

        if _profile is not None:
            t2 = time.perf_counter()
            _profile['infer_ms'] = (t2 - t1) * 1000.0

        # GPU output 파싱 (최적화 #1: 18개만 CPU 복사)
        result = self._postprocess_gpu(scale, pad_w, pad_h, h, w)

        if _profile is not None:
            t3 = time.perf_counter()
            _profile['postprocess_ms'] = (t3 - t2) * 1000.0

        return result

    def predict_bgra(self, bgra_image, _profile=None):
        return self.predict(bgra_image, is_bgra=True, _profile=_profile)

    # ── 후처리 (GPU 파싱) ────────────────────────────────────────────────────

    def _postprocess_gpu(self, scale, pad_w, pad_h, orig_h, orig_w):
        """GPU에서 top-1 찾고 keypoint 18개만 CPU로 복사"""
        result = type('PoseResult', (), {
            'keypoints_2d': {}, 'confidences': {},
            'detected': False, 'inference_time_ms': 0.0
        })()

        out = self._output_tensor[0]  # (300, 24) GPU tensor
        n_cand, n_feat = out.shape
        kpt_feats = self.num_kpts * 3  # 18

        # feature 레이아웃
        if n_feat == 4 + 2 + kpt_feats:    # 24: xyxy+conf+cls+kpts
            conf_idx = 4
            kpt_start = 6
        elif n_feat == 4 + 1 + kpt_feats:  # 23: xywh+conf+kpts
            conf_idx = 4
            kpt_start = 5
        else:
            return result

        # GPU에서 confidence 필터 + top-1 (CPU 복사 최소화)
        confs = out[:, conf_idx]  # (300,) GPU
        max_conf, best_idx = confs.max(dim=0)  # GPU에서 최대값

        # ⚠ Path C (2026-05-12) REVERTED: torch.cat + single .cpu() 가
        # 진정 *진정 *2 separate .item() + .cpu() 보다 *진정 *느림 (Jetson
        # 측정: 0.55ms → 0.94ms regression). torch.cat 의 *concat GPU op*
        # overhead 가 *.item()* 의 *single-element transfer 보다 expensive.
        # 진정 *교훈*: 가설 검증 의무 — premature optimization 잘못.
        if max_conf.item() < self.conf_thresh:
            return result

        # keypoint 18개만 CPU로 복사 (7200개 대신 18개)
        kpts_gpu = out[best_idx, kpt_start:kpt_start + kpt_feats]
        kpts = kpts_gpu.cpu().numpy().reshape(self.num_kpts, 3)

        # letterbox 역변환
        valid_count = 0
        for idx, name in self.KEYPOINT_MAP.items():
            if idx >= len(kpts):
                continue
            x, y, c = float(kpts[idx, 0]), float(kpts[idx, 1]), float(kpts[idx, 2])
            x = (x - pad_w) / scale
            y = (y - pad_h) / scale
            if c > 0.01 and 0 <= x <= orig_w and 0 <= y <= orig_h:
                result.keypoints_2d[name] = (x, y)
                result.confidences[name] = c
                if c > 0.3:
                    valid_count += 1

        if valid_count >= 3:
            result.detected = True
        return result

    # ── Depth Smoothing (최적화 #3) ──────────────────────────────────────────

    def get_3d_coords(self, result, depth_map, camera_intrinsics=None,
                      patch_size=5, alpha=0.7):
        """keypoint 2D → 3D 변환 + depth smoothing

        Args:
            result: predict() 반환값
            depth_map: ZED depth (float32, meters)
            camera_intrinsics: (fx, fy, cx, cy) 또는 None (pixel 좌표 + depth)
            patch_size: depth patch sampling 크기 (중앙값)
            alpha: temporal smoothing 계수 (0=이전만, 1=현재만, 0.7 추천)

        Returns:
            {joint_name: np.array([x, y, z])} 3D 좌표
        """
        if not result.detected or depth_map is None:
            return {}

        coords_3d = {}
        h, w = depth_map.shape[:2]
        r = patch_size // 2

        for name, (px, py) in result.keypoints_2d.items():
            ix, iy = int(round(px)), int(round(py))

            # patch sampling: 중앙값 depth (노이즈 제거)
            y1 = max(0, iy - r)
            y2 = min(h, iy + r + 1)
            x1 = max(0, ix - r)
            x2 = min(w, ix + r + 1)
            patch = depth_map[y1:y2, x1:x2]
            valid = patch[np.isfinite(patch) & (patch > 0)]

            if len(valid) == 0:
                continue

            z = float(np.median(valid))

            # 3D 좌표 계산
            if camera_intrinsics is not None:
                fx, fy, cx, cy = camera_intrinsics
                x3d = (px - cx) * z / fx
                y3d = (py - cy) * z / fy
            else:
                x3d, y3d = px, py

            current = np.array([x3d, y3d, z], dtype=np.float32)

            # temporal smoothing (이전 프레임과 보간)
            if name in self._prev_3d:
                current = alpha * current + (1 - alpha) * self._prev_3d[name]

            self._prev_3d[name] = current.copy()
            coords_3d[name] = current

        return coords_3d
