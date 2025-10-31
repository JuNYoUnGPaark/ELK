### 1. ELK Backbone

`ELKBlock`

- Structural Reparameterization(구조적 재매개변수화): Train/Inference 다르게 동작
- Train 시: 4개의 Conv1d + 1개의 Identity의 병렬 동작 (5개 branch)
- Inference 시: `reparmeterize()` 호출 → 5개의 branch를 1개의 Conv1d로 통합

`ELKBackbone`

- num_layers 만큼 `ELKBlock`을 쌓음.

### 2. Sequential Cross-Attention

`ImprovedSensorAttention` : 9개의 센서 Channel 간 관계 핛습

`TemporalCrossAttention` : 시간 순서 간 관계 학습

`SequentialCrossAttention` : 위 2가지를 순차 적용

### 3. Temporal Prototype Attention

- 모델이 학습한 특징 VS. 프로토타입 비교하여 최종 결정하는 모듈

`ClassConditionalTPA` : 각 Class별 전용 프로토타입 사용

### 4. Hybrid Model

`ELK_SequentialAttn_TPA` 

1. `ELKBackbone` 1차 특징 추출 
2. `SequentialCrossAttention` 특징 정제
3. `TPA` 정제된 특징으로 최종 분류 

### “Structural Reparameterization”

- **Training Mode**
    
    
    | kernel size | padding | BN |
    | --- | --- | --- |
    | 31 | 15 | +BN |
    | 29 | 14 | +BN |
    | 5 | 2 | +BN |
    | 3 | 1 | +BN |
    | IDENTITY |  | +BN |

다음의 CNN 통과 후 → 5개 branch 결과 더하기 → GELU activation → Conv1d + BN

- **Inference Mode**

→ 5개의 branch 합쳐진 1개의 Conv. → GELU activation → Conv1d + BN

### 전체 data flow

- x: (B, 9, 128)
- backbone: (B, 128, 128)
- permute: (B, 9, 128)
- 초기 예측: X = (B, 128), y = (B, 6)
- attn_1: (B, 128, 128)
- attn_2: z = (B, 128)
- 최종 예측: (B, 6)