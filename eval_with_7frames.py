import os
import torch
import cv2
import numpy as np
import h5py
import yaml
from glob import glob
from tqdm import tqdm
from piq import ssim, psnr, LPIPS

from core.models.VFPSIE import Model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def read_Image(path_to_image):
    image = cv2.imread(path_to_image, 1)  # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1)) / 255.0  # (C, H, W)
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

def read_Event(path_to_event):
    grid_voxel = h5py.File(path_to_event, 'r')['data'][:]
    return torch.tensor(grid_voxel, dtype=torch.float32).unsqueeze(0)

def compute_metrics(pred, gt):
    # 배치 차원을 유지 (B, C, H, W)
    
    # PSNR 계산
    psnr_value = psnr(pred, gt, data_range=1.0)
    
    # SSIM 계산
    ssim_value = ssim(pred, gt, data_range=1.0)
    
    # LPIPS 계산
    lpips_metric = LPIPS(replace_pooling=True)
    lpips_value = lpips_metric(pred, gt)
    
    return psnr_value.item(), ssim_value.item(), lpips_value.item()

def evaluate_sequence(sequence_path, model, device, skip_interval, predict_frames, dataset_type):
    image_dir = os.path.join(sequence_path, "images")
    event_dir = os.path.join(sequence_path, "events_hdf")

    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    event_paths = sorted(glob(os.path.join(event_dir, "*.hdf")))

    if len(image_paths) < 2 or len(event_paths) < 1:
        return None  # not enough data

    # 예측 결과를 저장할 디렉토리 생성
    seq_name = os.path.basename(sequence_path)
    output_dir = os.path.join(f"{dataset_type}_output", f"{predict_frames}frames", seq_name)
    os.makedirs(output_dir, exist_ok=True)

    results = []
    chunk_idx = 0
    
    with torch.no_grad():
        # skip_interval 개씩 건너뛰기
        total_chunks = len(event_paths) // skip_interval
        for i in tqdm(range(0, len(event_paths), skip_interval), desc=f"Processing {seq_name}", total=total_chunks):
            if i + predict_frames >= len(event_paths):
                break
                
            img0 = read_Image(image_paths[i]).to(device)
            psnr_list, ssim_list, lpips_list = [], [], []
            
            # predict_frames 개수만큼 예측
            for j in range(predict_frames):
                if i + j + 1 >= len(image_paths):
                    break
                    
                event = read_Event(event_paths[i + j]).to(device)
                pred = model(img0, event)
                gt_img = read_Image(image_paths[i + j + 1]).to(device)
                
                # 메트릭 계산
                psnr_val, ssim_val, lpips_val = compute_metrics(pred, gt_img)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                lpips_list.append(lpips_val)
                
                # 예측 이미지 저장
                image_name = os.path.basename(image_paths[i + j + 1])
                path_to_image = os.path.join(output_dir, image_name)
                # RGB에서 BGR로 변환하여 저장
                output_image = np.array(pred[0].permute(1, 2, 0).cpu().numpy() * 255.0)
                output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(path_to_image, output_image)
            
            # predict_frames 개수에 대한 평균 메트릭 저장
            if psnr_list:  # 리스트가 비어있지 않은 경우에만
                avg_psnr = np.mean(psnr_list)
                avg_ssim = np.mean(ssim_list)
                avg_lpips = np.mean(lpips_list)
                results.append((f"{seq_name}_{chunk_idx}", avg_psnr, avg_ssim, avg_lpips))
                chunk_idx += 1

    return results

def evaluate_all_sequences(data_root, model_ckpt_path, skip_interval, predict_frames, dataset_type):
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        raise RuntimeError("CUDA is not available. This model requires CUDA to run.")
    
    device = torch.device("cuda")
    model = Model().to(device)
    raw_ckpt = torch.load(model_ckpt_path, map_location='cpu')
    model.load_state_dict({k.replace("module.", ""): v for k, v in raw_ckpt.items()})
    model.eval()

    sequence_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')]

    all_results = []
    print("\n=== Evaluation Results ===")
    
    for seq_path in tqdm(sequence_dirs, desc="Evaluating sequences"):
        seq_name = os.path.basename(seq_path)
        print(f"\n- Evaluating: {seq_name}")
        results = evaluate_sequence(seq_path, model, device, skip_interval, predict_frames, dataset_type)
        if results:
            # 각 덩어리별 결과 출력
            for seq, psnr, ssim, lpips in results:
                print(f"{seq} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}")
            
            # 시퀀스별 평균 계산
            seq_psnr = np.mean([r[1] for r in results])
            seq_ssim = np.mean([r[2] for r in results])
            seq_lpips = np.mean([r[3] for r in results])
            print(f"\n{seq_name} Average - PSNR: {seq_psnr:.2f}, SSIM: {seq_ssim:.4f}, LPIPS: {seq_lpips:.4f}")
            print("-" * 60)
            
            all_results.extend(results)

    return all_results

if __name__ == "__main__":
    config = load_config("config.yaml")
    
    for dataset_type, dataset_config in config['datasets'].items():
        for dataset_path in dataset_config['paths']:
            dataset_name = os.path.basename(dataset_path)
            log_file = f"eval_results_{dataset_name}.log"
            
            print(f"\n{'='*50}")
            print(f"Evaluating dataset: {dataset_name}")
            print(f"Skip interval: {dataset_config['skip_interval']}, Predict frames: {dataset_config['predict_frames']}")
            print(f"{'='*50}\n")
            
            # 결과를 로그 파일로 저장
            with open(log_file, 'w') as f:
                # 표준 출력을 로그 파일로 리다이렉트
                import sys
                original_stdout = sys.stdout
                sys.stdout = f
                
                try:
                    evaluate_all_sequences(
                        dataset_path, 
                        "pretrained_model/VFPSIE.pth",
                        dataset_config['skip_interval'],
                        dataset_config['predict_frames'],
                        dataset_type
                    )
                
                finally:
                    # 표준 출력 복원
                    sys.stdout = original_stdout
            
            print(f"Results saved to {log_file}")