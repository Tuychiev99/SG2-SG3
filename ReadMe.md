# 자연스러운 이미지 생성 
헤어스타일 변경에서의 적용 가능성 검증  
생성 이미지의 품질, 일관성, 효율성 향상

#기존 StyleGAN2 기반 이미지 생성:
 위상 왜곡 발생 → 자연스러운 결과물 한계
복잡한 텍스처 표현의 어려움 
StyleGAN3 도입: 
 위상 유지로 품질 개선
 복잡한 이미지 생성 가능성 제고



## **SG3 코드 설명 및 작동 구조**

이 문서는 SG3 (StyleGAN3) 모델을 활용한 코드의 주요 부분을 설명합니다. SG3는 이미지의 복잡한 변환과 스타일 편집에 특화된 모델로, SG2 (StyleGAN2)보다 더 세밀하고 자연스러운 결과를 제공합니다.

### 1. **SG3 모델 로드**
SG3 사용 시, StyleGAN3 모델을 초기화하고 학습된 가중치를 로드합니다. `load_sg3_models()` 함수는 SG3 모델(`generator`), 옵션(`opts_sg3`), 세그멘테이션 모델(`seg`), 평균 이미지(`avg_img`)를 로드합니다. SG3는 더 복잡한 움직임과 시각적 요소를 처리하는 데 최적화되어 있으며, 이러한 점에서 SG2보다 강력한 성능을 발휘합니다.

#### 코드 예시:
```python
# SG3 모델 로드
from utils.model_utils import load_sg3_models

generator, opts_sg3, seg, avg_img = load_sg3_models(opts)
```

### 2. **Embedding (임베딩) 과정**
`Embedding_sg3` 클래스는 이미지 인버전을 수행합니다. 인버전은 입력 이미지를 StyleGAN의 잠재 공간에 맞게 변환하여 네트워크에서 해석 가능한 형태로 만드는 작업입니다. SG3의 인버전 과정은 특히 복잡한 이미지의 세부 정보를 더 잘 유지할 수 있도록 설계되어 있어, 보다 정교한 편집을 가능하게 합니다.

#### 코드 예시:
```python
# SG3 임베딩 과정
from scripts.Embedding import Embedding_sg3

re4e = Embedding_sg3(opts, generator)
src_latent = re4e.invert_image_in_W(image_path=src_image_path, device='cuda', avg_image=avg_img)
```

### 3. **편집 도구 및 프록시 사용**
SG3에서는 여러 편집 도구와 프록시를 사용하여 이미지를 변환합니다:
- **RefProxy_sg3**: 참조 이미지를 기반으로 스타일 변환을 수행합니다. 원본 이미지와 비슷한 스타일을 적용하기 위해 사용됩니다.
- **RefineProxy**: 스타일 편집 후 이미지를 추가적으로 정교화합니다. SG3는 이 과정에서 이미지의 세부적인 질감과 디테일을 더 잘 살립니다.
- **ColorProxy_sg3**: 색상 편집을 수행하며, 목표 색상 정보를 반영합니다. 
- **FaceEditor**: 특정 속성(예: 대머리 생성, 포즈 변경 등)을 조정합니다. SG3는 움직임이나 복잡한 변형을 처리하는 데 뛰어난 성능을 발휘합니다.

#### 코드 예시:
```python
# 편집 도구 및 프록시 사용
from scripts.ref_proxy import RefProxy_sg3
from scripts.refine_image import RefineProxy
from scripts.color_proxy import ColorProxy_sg3

ref_proxy = RefProxy_sg3(opts, generator, seg, re4e)
refine_proxy = RefineProxy(opts, generator, seg)
color_proxy = ColorProxy_sg3(opts, generator, seg)
```

### 4. **편집 과정**
편집 과정은 다음과 같이 이루어집니다:
1. **인버전**: 원본 이미지를 SG3 모델의 잠재 공간으로 인버전하여 잠재 벡터를 구합니다.
2. **스타일 적용**: `bald_proxy` 등 프록시를 통해 특정 스타일을 적용하고 편집된 잠재 벡터를 생성합니다.
3. **헤어스타일 특징 블렌딩**: `hairstyle_feature_blending_sg3()` 함수를 통해 헤어스타일 특징을 블렌딩하고 참조 이미지에 따라 잠재 공간을 생성합니다.
4. **정제 및 색상 적용**: `refine_proxy`와 `color_proxy`를 사용해 최종 결과를 정제하고 색상을 적용합니다.

#### 코드 예시:
```python
# 편집 과정 수행
from scripts.feature_blending import hairstyle_feature_blending_sg3

latent_global, visual_global_list = ref_proxy(global_cond, src_image=src_image, m_style=5, edit_latent=latent_bald)
blend_source, edited_hairstyle_img, edited_latent = hairstyle_feature_blending_sg3(generator, seg, src_image, input_mask, latent_bald, latent_global, avg_img)
final_image, blended_latent, visual_list = refine_proxy(blended_latent=edited_latent, src_image=src_image, ref_img=visual_global_list[-1], target_mask=target_mask)
visual_final_list = color_proxy(color_cond, final_image, blended_latent, blend_source)
```

### 5. **평가 지표**
코드의 마지막 부분에서는 SG3와 SG2로 생성된 이미지를 평가합니다. 평가에 사용된 주요 지표는 다음과 같습니다:
- **SSIM (Structural Similarity Index)**: 두 이미지 간의 구조적 유사성을 측정합니다. SG2가 SG3보다 조금 더 높은 점수를 보였으나, SG3는 복잡한 스타일 변환에서 더 자연스러운 결과를 제공합니다.
- **FFT Mean / FFT Std Dev**: 주파수 영역에서 이미지의 특성을 측정하는 지표입니다. SG3는 SG2와 비슷한 평균 값을 보였으나, 표준 편차에서 SG3가 더 낮아 상대적으로 일관된 주파수 특성을 가집니다. 이는 SG3가 더 자연스럽고 부드러운 결과를 생성함을 나타냅니다.
- **LPIPS (Learned Perceptual Image Patch Similarity)**: 두 이미지 간의 시각적 유사성을 측정하는 딥러닝 기반 지표로, SG3가 SG2보다 더 높은 값을 보이며, 이는 SG3가 더 복잡하고 풍부한 스타일 표현을 제공함을 의미합니다.
- **PSNR (Peak Signal-to-Noise Ratio)**: 원본 이미지와 재생성된 이미지 간의 신호 대 잡음 비율을 측정하며, SG2가 SG3보다 더 높은 값을 보였으나, SG3는 스타일 표현에 더 집중된 결과를 제공합니다.

#### 코드 예시:
```python
# SG2와 SG3 이미지 비교 평가
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
import numpy as np

# SSIM 계산
ssim_sg2 = ssim(src_img, sg2_img, data_range=src_img.max() - src_img.min())
ssim_sg3 = ssim(src_img, sg3_img, data_range=src_img.max() - src_img.min())

# LPIPS 계산
lpips_fn = LPIPS(net='alex')
lpips_sg2 = lpips_fn(src_img_tensor, sg2_img_tensor)
lpips_sg3 = lpips_fn(src_img_tensor, sg3_img_tensor)

# 결과 출력
print("SSIM (SG2 vs SG3):", ssim_sg2, ssim_sg3)
print("LPIPS (SG2 vs SG3):", lpips_sg2.item(), lpips_sg3.item())
```

### **종합적인 분석**
SG3 모델은 SG2에 비해 복잡한 스타일 변환, 색상 조정, 움직임 표현에서 강점을 가지고 있습니다. 특히 LPIPS 지표에서 SG3의 높은 점수는 시각적으로 더 풍부하고 세밀한 변환을 수행할 수 있음을 나타냅니다. SG3는 디테일과 스타일 표현력이 뛰어나며, 복잡한 이미지 편집에 적합한 모델입니다. SSIM 및 PSNR 측면에서 SG2가 원본과의 유사성을 조금 더 잘 유지하고 있지만, SG3는 더 자연스럽고 다채로운 결과를 생성하는 데 강점을 지니고 있습니다.

