import numpy as np
import torch
import torch.nn.functional as F

def post_process(c, size_list, outputs_list):
    print('Multi-scale sizes:', size_list)
    # 初始化累加器
    logp_sum = None
    prop_sum = None

    print(f"outputs_list 长度: {len(outputs_list)}")  
    print("batch_size",c.batch_size)

    with torch.no_grad():
        for i,outputs in enumerate(outputs_list):
            # 处理每个尺度的输出
            outputs = torch.cat(outputs, 0).float()
            # 计算logp_map (增量式处理)
            current_logp = F.interpolate(
                outputs.unsqueeze(1),
                size=c.input_size,
                mode='bilinear',
                align_corners=True
            ).squeeze(1)
            # 计算prob_map (增量式处理)
            output_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
            current_prob = torch.exp(output_norm)
            current_prop = F.interpolate(
                current_prob.unsqueeze(1),
                size=c.input_size,
                mode='bilinear',
                align_corners=True
            ).squeeze(1)
            # 增量累加
            if logp_sum is None:
                logp_sum = current_logp
                prop_sum = current_prop
            else:
                logp_sum += current_logp
                prop_sum += current_prop
            # 立即释放不再需要的中间变量
            del outputs, output_norm, current_prob, current_logp, current_prop
            torch.cuda.empty_cache()
        # 最终计算
        # 1. 计算mul结果
        logp_sum -= logp_sum.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        prop_map_mul = torch.exp(logp_sum)
        anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
        # 2. 计算add结果
        prop_map_add = prop_sum.cpu().numpy()
        anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add
        # 3. 计算anomaly_score (分块处理避免大矩阵操作)
        batch_size = anomaly_score_map_mul.shape[0]
        top_k = int(c.input_size[0] * c.input_size[1] * c.top_k)
        # 分块计算anomaly_score
        chunk_size = 16  # 可根据显存调整
        anomaly_scores = []
        for i in range(0, batch_size, chunk_size):
            chunk = anomaly_score_map_mul[i:i+chunk_size]
            scores = np.mean(
                chunk.reshape(len(chunk), -1).topk(top_k, dim=-1)[0].cpu().numpy(),
                axis=1
            )
            anomaly_scores.append(scores)
            del chunk
            torch.cuda.empty_cache()
        anomaly_score = np.concatenate(anomaly_scores)
        # 最后转换mul结果到numpy
        anomaly_score_map_mul = anomaly_score_map_mul.cpu().numpy()
        # 释放剩余变量
        del logp_sum, prop_sum, prop_map_mul
        torch.cuda.empty_cache()
    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul