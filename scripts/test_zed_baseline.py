"""ZED baseline test — single thread, just grab + retrieve. ~5초.

목적: kill_test 가 silent exit 하는 원인을 *threading issue* vs
*ZED Python binding 기본 issue* 분리.

작동 시 → kill_test 의 threading 이 issue (다음 단계 진단)
미작동 시 → ZED Python binding 의 기본 issue (config 문제)

사용법:
    sudo PYTHONPATH=/home/chobb0/.local/lib/python3.10/site-packages \\
        python3 -u scripts/test_zed_baseline.py 2>&1 | tee /tmp/test_baseline.log
    echo "python_exit=${PIPESTATUS[0]}"
"""
import sys
import time
import traceback

print("opening ZED...", flush=True)
import pyzed.sl as sl

zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.SVGA
init.camera_fps = 120
init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init.coordinate_units = sl.UNIT.METER

status = zed.open(init)
print(f"ZED open: {status}", flush=True)
if status != sl.ERROR_CODE.SUCCESS:
    print("FAIL — ZED open", flush=True)
    sys.exit(1)

rt = sl.RuntimeParameters()
image_mat = sl.Mat()
depth_mat = sl.Mat()

print("starting 600 frame loop (~5s at 120fps)...", flush=True)
ok_count = 0
grab_fail_count = 0
exception_count = 0
t_start = time.time()

for i in range(600):
    try:
        grab_result = zed.grab(rt)
    except Exception as e:
        print(f"  frame {i} grab() exception: {e}", flush=True)
        traceback.print_exc()
        exception_count += 1
        time.sleep(0.001)
        continue

    if grab_result != sl.ERROR_CODE.SUCCESS:
        grab_fail_count += 1
        if grab_fail_count <= 5:
            print(f"  frame {i} grab fail: {grab_result}", flush=True)
        time.sleep(0.001)
        continue

    try:
        ts_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
    except Exception as e:
        print(f"  frame {i} get_timestamp exception: {e}", flush=True)
        traceback.print_exc()
        exception_count += 1
        break

    try:
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    except Exception as e:
        print(f"  frame {i} retrieve_image exception: {e}", flush=True)
        traceback.print_exc()
        exception_count += 1
        break

    try:
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    except Exception as e:
        print(f"  frame {i} retrieve_measure exception: {e}", flush=True)
        traceback.print_exc()
        exception_count += 1
        break

    ok_count += 1

    if i % 100 == 0 or i < 5:
        print(f"  frame {i} ok, ts={ts_ns}, elapsed={time.time()-t_start:.2f}s",
              flush=True)

elapsed = time.time() - t_start
print(f"\nDONE: ok={ok_count}, grab_fail={grab_fail_count}, exception={exception_count}",
      flush=True)
print(f"  elapsed={elapsed:.2f}s, throughput={ok_count/elapsed:.1f}Hz", flush=True)

zed.close()
print("ZED closed.", flush=True)

if ok_count >= 500:
    print("\n=== BASELINE PASS ===", flush=True)
    print("→ ZED 기본 작동 OK. kill_test 의 threading issue 진단 다음 단계.", flush=True)
    sys.exit(0)
else:
    print(f"\n=== BASELINE FAIL ({ok_count}/600) ===", flush=True)
    print("→ ZED Python binding 기본 issue. config 또는 환경 검증 필요.", flush=True)
    sys.exit(1)
