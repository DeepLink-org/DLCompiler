import subprocess

def is_gpu_idle(gpu_id):
    try:
        # 调用 nvidia-smi 命令
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
            f"--id={gpu_id}"
        ])
        # 解析输出
        processes = result.decode('utf-8').strip().split('\n')
        # import pdb; pdb.set_trace()
        # 检查是否有进程正在运行
        if len(processes) == 1 and processes[0] == '':
            return True  # GPU is idle
        else:
            return False  # GPU is not idle
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to check GPU status.")
    except Exception as e:
        raise RuntimeError("Failed to check GPU status:{e}")

def get_idle_device():
    for gpu_id in range(8):
        if is_gpu_idle(gpu_id) == True:
            print(f"GPU {gpu_id} is idle, we will use cuda:{gpu_id}")
            return f"cuda:{gpu_id}"
    print("[WARN] All GPU device is busy, performance data maybe inaccurate.")
    return "cuda"
