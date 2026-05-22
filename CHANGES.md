# PekoraNet v3 — 전체 버그 수정 내역

---

## [FIX A] AdaptiveAvgPool2d 제거 — `pekora_encoder.py`

### 문제
```python
self.pool = nn.AdaptiveAvgPool2d((8, 8))  # 32x32 → 8x8
```
Global Average Pooling이 COST2100 indoor 채널의 핵심인
앵귤러-딜레이 도메인 **공간 위치 정보를 소멸**시킴.

### 수정
Conv(2→32) × 2 → Flatten → Linear(32768 → latent_dim)
Pool 없이 공간 정보를 그대로 FC로 전달.

---

## [FIX B] 진짜 병목(Bottleneck) 구조 적용 — `pekora_encoder.py`, `train.py`

### 문제
```python
# 원본: encoder는 항상 512 출력
# CR=1/4: L=512, latent[:, :512] → 전부 통과, truncation 없음
# → 병목이 없어 압축 학습 자체가 불가능
```

### 수정
CR 인자를 받아 encoder/decoder를 해당 `latent_dim`으로 직접 생성:
```python
latent_dim = CR_MAP[args.cr]   # 예: CR=1/4 → 512, CR=1/16 → 128
encoder = PekoraEncoder(latent_dim=latent_dim)
decoder = PekoraDecoder(latent_dim=latent_dim)
```
입력 2048 → latent_dim으로 실제 압축 발생.

---

## [FIX C] Importance-based Sorting 제거 — `pekora_encoder.py`

### 문제
```python
importance = self.importance_head(latent)
score = torch.abs(latent) * importance
idx = torch.argsort(score, dim=1, descending=True)
ordered = torch.gather(latent, 1, idx)
```
argsort 기반 정렬은 미분 불가 연산으로 gradient 흐름을 불안정하게 만듦.
reconstruction loss와 목표가 충돌.

### 수정
단순 FC 병목으로 자연스러운 특징 선택 학습.

---

## [FIX D] Decoder 출력 활성화 수정 — `pekora_decoder.py`

### 문제
```python
return torch.tanh(x)   # 출력 [-1, +1], 타겟 [0, 1] — 불일치
```

### 수정
```python
return torch.sigmoid(x)   # 출력 [0, 1] — 타겟과 일치
```

---

## [FIX E] EEV 모듈 분리 — `train.py`

### 문제
EEVEstimator에 `torch.randn` 랜덤 입력 → 채널 정보와 무관.
latent에 노이즈가 concat되어 복원 방해.

### 수정
기본 복원 학습에서 EEV 모듈 완전 분리.
(EEVEstimator 파일은 유지, 추후 재도입 가능)

---

## [FIX G] 데이터 전처리 [0,1] 유지 — `utils/dataset.py`

### 근거
CsiNet 논문은 [0,1] 데이터로 NMSE 계산 → -20 dB 달성.
`H - 0.5`로 DC 제거 시 NMSE 분모가 달라져 논문과 직접 비교 불가.
[0,1] 그대로 유지해야 CsiNet 비교 기준이 동일해짐.

---

## [FIX H] ProgressiveOrderingLoss 제거 — `train.py`, `utils/losses.py`

### 문제
```python
loss = reconstruction_loss + 0.05 * ordering_loss
```
ordering_loss가 reconstruction_loss와 목표 충돌.
epoch 1→2에서 Loss 86% 급락 후 즉시 수렴하는 원인.

### 수정
순수 MSELoss만 사용:
```python
loss = criterion(recon, batch)
```

---

## [FIX I] NMSE 로그 CR 수 나누기 — `train.py`

`--cr all` 모드 시 CR 4개분 NMSE가 누적되는 버그 수정.
현재 v3는 `--cr`을 단일 CR로만 받아 이 문제를 구조적으로 제거.

---

## 변경 파일 요약

| 파일 | 주요 수정 |
|------|-----------|
| `models/pekora_encoder.py` | Pool 제거, 진짜 bottleneck, sorting 제거 |
| `models/pekora_decoder.py` | EEV 제거, sigmoid 출력, 단순 구조 |
| `train.py`                 | CR별 모델 생성, OrderingLoss 제거, EEV 제거 |
| `evaluate.py`              | CR별 checkpoint 로드 |
| `utils/dataset.py`         | [0,1] 유지 (CsiNet 표준) |
| `utils/losses.py`          | ProgressiveOrderingLoss 제거 |
| `utils/metrics.py`         | torch/numpy 버전 NMSE 분리 |

---

## 사용법

```bash
# CR=1/4 indoor 학습
python train.py --scenario indoor --cr 1/4 --epochs 1000 --amp

# CR=1/16 학습
python train.py --scenario indoor --cr 1/16 --epochs 1000 --amp

# 평가
python evaluate.py --scenario indoor --cr 1/4
```

## 기대 학습 곡선 (CR=1/4, indoor)

| Epoch | 예상 NMSE(dB) |
|-------|--------------|
| 1     | -10 ~ -15    |
| 10    | -15 ~ -17    |
| 100   | -18 ~ -19    |
| 1000  | -20 이하      |
