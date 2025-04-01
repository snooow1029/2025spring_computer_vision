import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1  # 定義窗口大小
        self.pad_w = 3 * sigma_s  # Padding 大小
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, 
                                        self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, 
                                            self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        x = np.arange(-self.pad_w, self.pad_w + 1)
        X, Y = np.meshgrid(x, x)
        s_kernel_2D = np.exp(-0.5 * (X**2 + Y**2) / self.sigma_s**2)
        r_kernel = np.exp(-0.5*(np.arange(256)/255)**2/self.sigma_r**2)
        
        wgt_sum, result = np.zeros(padded_img.shape), np.zeros(padded_img.shape)
        
        # 預先計算所有偏移的組合
        offsets = []
        spatial_weights = []
        
        for y in range(-self.pad_w, self.pad_w + 1):
            for x in range(-self.pad_w, self.pad_w + 1):
                offsets.append((y, x))
                spatial_weights.append(s_kernel_2D[self.pad_w + y, self.pad_w + x])
        
        # 對每個偏移組合進行一次計算
        for i, ((y, x), s_w) in enumerate(zip(offsets, spatial_weights)):
            shifted_guidance = np.roll(padded_guidance, [y, x], axis=[0, 1])
            guidance_diff = np.abs(shifted_guidance - padded_guidance).astype(np.uint8)
            
            if guidance_diff.ndim == 2:  # 灰階
                r_w = r_kernel[guidance_diff]
            else:  # 彩色
                r_w = np.prod([r_kernel[guidance_diff[:,:,c]] for c in range(guidance_diff.shape[2])], axis=0)
        
            t_w = s_w * r_w

            shifted_img = np.roll(padded_img, [y, x], axis=[0, 1])
            
            # 根據圖像維度適當更新結果
            if padded_img.ndim == 2:  # 灰階圖像
                result += shifted_img * t_w
                wgt_sum += t_w
            else:  # 彩色圖像
                for c in range(padded_img.shape[2]):
                    result[:,:,c] += shifted_img[:,:,c] * t_w
                    wgt_sum[:,:,c] += t_w
                
        output = (result / wgt_sum)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, :]
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    # def _print_timing_analysis(self, timing):
    #     """打印時間分析結果"""
    #     print("\n====== 執行時間分析 ======")
    #     print(f"總執行時間: {timing['total_time']:.4f} 秒")
    #     print("\n各部分時間佔比:")
        
    #     # 計算各部分佔總時間的百分比並排序
    #     percentages = {k: (v / timing['total_time']) * 100 for k, v in timing.items() if k != 'total_time'}
    #     sorted_times = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        
    #     for name, percentage in sorted_times:
    #         print(f"{name}: {timing[name]:.4f} 秒 ({percentage:.2f}%)")
        
    #     print("==========================\n")