# RedLamp DS 작업 컨텍스트 (새 Claude 세션 인수인계용)

이 문서는 RedLamp_Check 저장소(`/Users/sokim/Desktop/CoreModel/Git/RedLamp`)에서 진행 중인
"DS" 분석 작업 전체(Experiment_1/2 → DS_0 → DS_1 → DS_2 → DS_3)와, 가장 최근 단계인
DS_3(정성적 진단 플롯)의 배경/현재 상태를 정리한 것이다. 새로운 Claude 세션에 이어서 작업을
시킬 때 이 문서를 먼저 읽히면 된다.

**이 문서 작성자(나, Claude)가 확실히 아는 내용과, 대화 중 짧게 확인만 하고 자세히는 파지
않은 내용을 구분해서 적었다.** "확인 필요" 표시가 붙은 부분은 내가 추측하지 않고, 어디를
보면 알 수 있는지만 적어뒀다 — 새 세션도 그 경로를 열어보면 나와 동일한 정보를 얻을 수 있다.

## 1. 전체 프로젝트 배경

`/Users/sokim/Desktop/CoreModel/Git/` 아래 형제 저장소 3개로 작업 중:

- **RedLamp_Check** (`RedLamp`, github.com/skimrepo/RedLamp_Check) — arXiv:2505.20765 RedLamp
  논문 재현 코드 기반. `main.py`에 핵심 모델(`ConvAEC`)과 손실/스코어링 로직이 있다.
- **Core-Clustering** — RedLamp의 데이터 로딩/윈도잉/주입 로직을 재구현·재사용하는 별도
  파이프라인(`core_clustering/` 패키지). AnomSim 데이터를 학습에 쓸 수 있게 다리 역할.
- **AnomSim** — 합성(synthetic) 이상치 시계열 생성기. `anomsim.anomalies.base.get_anomaly`
  레지스트리에 11종 이상치 타입(RedLamp와 동일한 타입: spike, flip, speedup, noise, cutoff,
  average, scale, wander, contextual, upsidedown, mixture)이 구현되어 있다.

목표는 RedLamp 논문 재현을 넘어서, "Self"(entity별 자기 학습 모델)와 "Cross-AnomSim"(AnomSim
합성 데이터로 학습한 범용 모델)을 UCR anomaly archive 실제 데이터에 대해 비교/진단하는 것.
작업은 `result/Experiment_1`, `Experiment_2`, `DS_0`, `DS_1`, `DS_2`, `DS_3` 순으로 단계적으로
쌓여왔다 (각 폴더가 `result/` 아래 존재).

## 1.5. 단계별 실험/DS 히스토리 (Experiment_1/2, DS_0~DS_2)

`result/` 아래에 단계별로 폴더가 쌓여 있다: `result/result/Experiment_1`, `result/Experiment_2`,
`result/DS_0`, `result/DS_1`, `result/DS_2`, `result/DS_3` (이 세션 이전에 만들어졌고, 나는
대화 중 필요할 때마다 부분적으로 확인했을 뿐 전체를 처음부터 설계하진 않았다).

**Experiment_1** (확인함): `result/result/Experiment_1/Models/{ModelType}/{seed}/bestmodel.pkl`
형태로 모델 체크포인트 저장. 모델 타입은 최소 3종: **Self**(UCR entity 하나당 자기 데이터로만
학습한 모델), **Cross-OpenSource**, **Cross-AnomSim**(AnomSim_v1 144개 entity를 풀링해서 학습한
범용 모델). 이 세션에서 주로 다룬 건 Self와 Cross-AnomSim 둘의 비교. 학습 시 검증은 "injected
pseudo-anomaly"(=UCR validation set에 인위적으로 anomaly 타입을 주입해서 만든 가짜 정답)로
한다 (Experiment_2 README에서 대조 설명함). 학습 스크립트는 아마 `main.py`(Self, 직접 실행)와
`scripts/domain_generalization.py`/`scripts/continuous_pool_scaling.py`(Cross 계열) 근처일
것으로 보이나, 정확한 실행 커맨드/스크립트 매핑은 **확인 필요** — `scripts/run_multiseed_training.py`,
`scripts/cross_inference.py`를 먼저 보면 됨.

**Experiment_2** (확인함, README 있음: `result/Experiment_2/Results/README.txt`): Experiment_1과
**같은 모델 체크포인트**를 재사용해서, 이번엔 **진짜 ground-truth anomaly**(UCR/KPI(iops) 실제
테스트셋 라벨)로 다시 채점한 것. 지표는 VUS_ROC, VUS_PR, R_AUC_ROC, R_AUC_PR, RF(+ UCR 전용
peak_in_range). RF 지표 계산 방식은 TSB_UAD 기준으로 이 세션 초반에 상세히 설명함(existence
reward + weighted overlap reward 기반 Range-Recall/Range-Precision의 조화평균, threshold =
mean(score)+3*std(score)). AnomSim_v1은 real label이 없어서 Experiment_2엔 안 나옴.

**DS_0** (파일 목록만 확인, 세부 로직은 확인 필요): `result/DS_0/{dataset}/{entity}/` 아래
`anomaly_example.png`, `tsne.png`, `waveform.png` 3개 파일이 entity마다 있음 (dataset은
anomaly_archive/anomsim_v1/iops/msl/smap/smd). 초기 EDA 단계로 보임 — waveform 시각화,
anomaly 예시, (아마 분류기 임베딩의) t-SNE. 생성 스크립트 이름은 **확인 필요**.

**DS_1** (CSV 몇 개를 이 세션에서 열어봄): `result/DS_1/entity_metadata.csv`(entity별
category/train_end/test_length/anomaly_length/`groups`/`gap_exp1_bad`/`gap_exp1_good`/
`gap_exp2_bad`/`gap_exp2_good` 컬럼), `result/DS_1/group_summary.csv`(exp1_bad n=8,
exp1_good n=10, exp2_bad n=15, exp2_good n=15 그룹별 length/anomaly_length 통계),
`result/DS_1/type_confusion.csv`(entity×anomaly_type×model별 분류 정확도). 즉 DS_1은 Experiment_1
/2 성능 기준으로 UCR entity들을 "good"/"bad" 그룹으로 나누고(무슨 지표의 "갭"으로 나눈 것으로
보임 — 컬럼명이 `gap_exp1_bad`처럼 "gap"), 그 그룹별로 길이/타입별 분류정확도를 비교하는
단계로 보임. **이 세션 시작 직전(요약되어 안 보이는 부분)에 "정상구간 점수/불량구간 점수/갭을
그룹 평균으로, bad/good 외에 전체/partial-전체로도 구해달라"는 요청과 "group_summary_table.csv"
언급이 있었는데, 이게 정확히 `group_summary.csv`를 가리키는지 다른 파일인지, "갭"의 정확한
계산식이 뭔지는 확인 필요** — `scripts/analyze_ds1_gap_entities.py`를 먼저 열어보면 나올
가능성이 높음(파일명으로 추정, 이전 대화에서 스크립트 목록 grep 결과 존재 확인됨).

**DS_2** (README 없음, 스크립트에서 역추적함): `scripts/analyze_score_oscillation.py` +
`scripts/run_score_oscillation_parallel.py`가 `result/DS_2/oscillation/oscillation_metrics.csv`
생성 — TSB_UAD의 `get_metrics`, anomaly score의 rolling std/correlation 등으로 "score
oscillation"(스코어가 얼마나 들쭉날쭉한지)을 정량화하는 것으로 보임. `result/DS_2/achievability`,
`result/DS_2/reference_distance_metrics.csv`도 있는데 **정확한 의미/계산 방식은 확인 필요**
(전자는 "달성 가능한 성능의 상한" 같은 걸 재는 것으로 추정, 후자는 아마
`scripts/plot_reference_distance_comparison.py`와 관련 — 스크립트를 직접 열어봐야 확실함).

**DS_3**: 아래 3~7절에서 상세히 다룸(이 세션의 메인 작업).

## 2. 핵심 재사용 함수 (절대 재구현하지 말고 이거 써야 함)

- `main.mse(input, pred, mean=True)` — 윈도우 전체에 대한 raw MSE (스무딩/정규화 이전).
- `main.anomaly_scoreing(input, pred, pred_label, threshold=0.05, return_components=False)` —
  `mse_score`(재구성오차 기반)와 `ce_score`(분류기 novelty, "흔히 예측되지 않는 클래스에
  얼마나 확률을 줬는지") 를 각각 `main.convolve_minmax_score`(박스컨볼루션 스무딩 +
  [0,1] min-max 정규화)로 처리한 뒤 50:50으로 섞어서 최종 `score`를 만든다.
  `return_components=True`로 부르면 `(score, mse_score, ce_score)` 3개를 받을 수 있다
  (하위호환 유지된 채로 이 세션 이전에 추가됨).
- `main.ConvAEC` — 오토인코더+분류기 모델. `.eval()` 필수.
- `loaders/loader_aug.py`의 `Loader_aug` (RedLamp 쪽 학습 데이터 증강/주입기),
  `Loader_aug.select_anomalies(anomaly_type, Y, window_start, window_end)` — 특정 타입의
  이상치를 특정 윈도우에 주입하는 실제 로직(11종 각각 `_inject_spike` 등 private 메서드).
- Core-Clustering `core_clustering/online_dataset.py`의 `OnlineWindowedDataset` — AnomSim
  데이터를 학습용으로 윈도잉+주입(같은 11종, `anomsim.anomalies.base.get_anomaly` 통해).
- `scripts/full_reproduction_metrics.py`의 `score_entity(...)` — UCR test set 전체에 대해
  dense window_step=1로 한 번 훑어서 `mse_score/ce_score/score/reconstruction/real_labels`
  등을 계산하고 `.npz` 캐싱하는 함수. 앞쪽 `window_size-1`개는 0으로 패딩(윈도우가 아직
  안 찼으니 무의미).

## 3. DS_3의 목적: "4-6패널 진단 플롯"

DS_0~2가 수치적(정량적) 비교였다면, DS_3은 **정성적** 진단이다. 특정 entity의 특정 구간에서
raw 신호/모델 재구성/여러 스코어를 timestep 축으로 겹쳐서 보는 플롯을 대량으로 생성한다.
공유 라이브러리는 `scripts/local_diagnostic_curves.py`이고, 이걸 쓰는 3개 스크립트가 있다:

- `scripts/build_self_train_val_diagnostics.py` — Self 모델, UCR entity들의 train/val split
- `scripts/build_anomsim_train_val_diagnostics.py` — Cross-AnomSim 모델, AnomSim_v1 데이터
- `scripts/build_ucr_test_diagnostics.py` — 두 모델 다, UCR test set (진짜 ground-truth anomaly)

각각 병렬 실행용 오케스트레이터가 있다 (`run_self_train_val_diagnostics_parallel.py`,
`run_anomsim_train_val_diagnostics_parallel.py`, `run_ucr_test_diagnostics_parallel.py`, 그리고
DS_2용 `run_score_oscillation_parallel.py`도 있음). 전부 `--shard_index`/`--num_shards`로
entity 리스트를 나눠 여러 프로세스로 돌리고, 서브프로세스마다 `OMP_NUM_THREADS` 등을
`cores // num_shards`로 캡을 걸어 스레드 오버서브스크립션을 막는다 (서버가 256코어인데
--num_shards 32로 돌렸을 때 전부 100%로 차서 발견/수정한 버그).

## 4. 플롯 한 페이지가 만들어지는 방식 (가장 최근 재설계, 중요)

**설계 원칙**: 페이지 하나 = 독립된 작은 로컬 실험.

1. focus window 하나(`window_size`, UCR은 보통 100)를 정한다.
2. 그 앞뒤로 실제(원본) 데이터를 `window_size*2`씩 붙여서 총 `5*window_size` 길이의 통짜
   로컬 시계열을 만든다 (`local_diagnostic_curves.build_local_chunk`).
3. Train/Val 페이지는 focus window **안에만** 해당 타입의 이상치를 한 번(랜덤 위치/크기로)
   주입한다 — RedLamp는 `Loader_aug.select_anomalies`, AnomSim은 `get_anomaly(type)().apply()`
   를 그대로 재사용(주입 로직 자체는 절대 새로 작성하지 않음). 앞뒤 context는 원본 그대로.
   Test 페이지는 주입 없음 — 실제 UCR 신호 그대로.
4. 이 5W짜리 로컬 시계열 전체에 `window_step=1` dense pass를 돌려서(`dense_windows_from_chunk`
   로 슬라이딩 윈도우 스택 만들고, `compute_dense_curves`로 모델 통과) reconstruction/mse/ce/
   score를 뽑는다. 각 윈도우의 값은 그 윈도우의 **마지막 timestep**에 배정(겹치는 dense
   윈도우들을 이렇게 정렬하는 게 기존 `score_entity`의 컨벤션 그대로).

**6개 패널**: 1)raw+reconstruction(회색=context/파란=focus), 1.25)`|raw-reconstruction|`
pointwise 절대오차(정규화 없음), 1.5)MSE(raw) 윈도우평균오차(정규화 없음), 2)MSE_Norm_Smooth,
3)CE_Norm_Smooth, 4)Anomaly_Norm_Smooth (2~4는 스무딩+0~1 정규화, y축 고정).

**중요한 최근 변경**: 2~4번 패널의 0~1 정규화가 이제 "이 페이지의 로컬 청크" 기준이다
(예전엔 entity 전체 split 기준이어서 국지적으로 튀는 부분이 바닥에 눌려 안 보이는 문제가
있었음 — 이번 재설계로 부수적으로 해결됨). 페이지 **끼리는** 2~4번을 직접 비교하면 안 되고
(서로 다른 정규화 기준), 절대 비교가 필요하면 1.25/1.5번(정규화 없음)을 봐야 한다.

**성능**: 원래 (entity, split, type)당 한 번의 거대한 whole-split dense pass를 돌려서 5개
샘플 페이지를 "공짜로" 슬라이싱하는 설계였는데, 사용자가 "focus window+context를 개별
시계열처럼 취급하고 싶다"고 재설계를 요청해서 지금 방식으로 바뀜. 대신 모델 forward pass는
`compute_dense_curves_batch`로 한 split의 최대 60개(type×sample) 조합을 한 번의 배치 호출로
묶어서 처리(Python/dispatch 오버헤드 절감, 수치적으로 개별 호출과 완전히 동일함을 테스트로
확인함). UCR test는 포지션 선정(진짜 anomaly 구간 + Self/Cross의 mse/ce argmax/argmin, 최대
9개/entity, 겹치면 병합)은 여전히 whole-split `score_entity` 결과를 쓰고, 그림 그릴 때만
로컬 재계산한다.

**캐싱**: entity당, split당, type당, sample당 `.npz` 하나(`{entity}_{split}_{type}_s{i}.npz`
형식). Train/val PDF는 entity당 1개 파일(`Self_Train_{entity}.pdf` 등), 12섹션(Normal+11타입)
×5샘플=60페이지. Test는 entity당 최대 9페이지, `--num_shards`로 쪼갤 때 파일별로 나뉘어서
병합 단계가 필요 없음(entity가 겹치지 않으므로).

## 5. 이번 세션에서 고친 실제 버그들 (참고용)

- `main.py`의 `Trainer.validation()`이 배치 손실을 합산만 하고 평균을 안 내던 버그(train()은
  올바르게 나눔) — 고쳐서 커밋함. entity별 best-epoch 선택엔 영향 없었지만 cross-entity 비교는
  무의미했었음.
- `main.py`의 `test()`/`extract_embeddings()`가 `torch.load()`에 `map_location=device`를 안
  줘서, CPU로 강제해도 체크포인트가 저장됐던 원래 디바이스(CUDA)로 먼저 로드를 시도하던 버그.
- `build_ucr_test_diagnostics.py`가 유일하게 GPU(`--gpu 0`)를 쓰고 있었는데, 32개 shard가
  동시에 같은(다른 job들이 78GB 점유 중인) GPU에 CUDA context를 잡으려다 OOM. CPU로 고정해서
  해결(다른 두 스크립트는 원래부터 CPU).
- `plot_diagnostic_page`에서 `ax.axvspan()`으로 real anomaly 구간을 표시할 때, matplotlib이
  그 사각형도 autoscale에 포함시켜서, focus window와 멀리 떨어진 anomaly가 있으면 x축이
  그 둘을 다 포함하도록 늘어나 실제 신호가 한쪽 구석에 찌그러지는 버그. `ax.set_xlim(d0,d1)`
  명시적으로 고정해서 해결.
- Panel 1에서 context/focus 경계에 직선 아티팩트(boolean masking으로 두 조각이 필터링된
  배열에서 인접해져서 matplotlib이 실제로는 없는 구간을 직선으로 이어 그림) — context를
  먼저 통짜로 그리고 focus를 위에 덧그리는 방식으로 해결.

## 6. 작업 컨벤션 (반드시 지켜야 함)

- 커밋/푸시는 사용자가 명시적으로 요청했을 때만 ("부탁해", "푸시해줘" 등). 커밋 메시지는
  HEREDOC으로 작성하고 끝에 `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
  매 커밋마다 새로 만들고(amend 금지, 명시적 요청 없이는).
  파괴적 git 명령(force push, reset --hard 등)은 명시적 요청 없이 절대 안 함.
- 기존 함수 재사용 우선 — `main.mse()`, `main.anomaly_scoreing()`, `Loader_aug.select_anomalies`,
  `get_anomaly()`, `ci.discover_entity`, `frm.discover_dataset_entities` 등. 새 지표/새 주입
  로직을 임의로 만들지 말고 있는 걸 재사용.
- 새로 만드는 "전체 스케일" 스크립트는 항상 (a) `.npz` 캐싱으로 재개 가능해야 하고
  (`--force`로 우회 가능), (b) `--shard_index`/`--num_shards`로 병렬화 가능해야 한다
  (`analyze_score_oscillation.py`/`run_score_oscillation_parallel.py` 패턴 참고).
- 서버는 코어가 아주 많음(256개 근처, 사용자 로컬 환경은 10코어 맥). 병렬 스크립트는 서브
  프로세스별 OMP/MKL 스레드 수를 명시적으로 캡 걸어야 오버서브스크립션이 안 남.
- 사용자는 한국어로 대화하며, 스크린샷으로 실제 플롯 결과를 보여주고 구체적인 시각적 버그를
  지적하는 방식으로 피드백을 준다 (예: "이 사이 직선이 이상해", "패널2가 바닥에 붙어있어").
  이런 지적은 대부분 진짜 버그였음 — 가볍게 넘기지 말고 근본 원인을 찾아서 고칠 것.

## 7. 현재 상태 / 다음에 할 일

- 위 재설계(로컬 청크 방식 + 배치 추론)는 로컬에서 실제 데이터(AnomSim 실제 체크포인트 +
  실제 데이터)와 합성 mock 데이터로 검증 완료, 커밋 `cf133be`로 커밋+푸시됨(origin/main과
  일치 확인됨).
- 사용자가 서버에서 캐시(`result/DS_3/curves_cache/{self,anomsim,test}`) 삭제 후 3개
  오케스트레이터를 전체 entity에 대해 다시 돌릴 예정. 명령어는 이 문서 작성 직전 대화에서
  전달함 (`run_self_train_val_diagnostics_parallel.py --run_name test --num_shards 32` 등,
  3개를 순서대로 돌리는 걸 추천했음 — 동시에 돌리면 코어 예산이 오버라이드가 안 돼서 대략
  3배 오버서브스크립션이 생기지만 심각한 수준은 아님).
- 다음 단계로 예상되는 것: 사용자가 서버 결과(PDF들)를 다운받아서 스크린샷으로 추가 시각적
  버그/이상한 패턴을 지적할 가능성이 높음 (지금까지 패턴이 그래왔음). 그때마다 근본 원인을
  찾아서(추측하지 말고 실제 코드/데이터로 검증) 고치고, 로컬에서 재현 가능한 범위에서
  검증 후 커밋.
