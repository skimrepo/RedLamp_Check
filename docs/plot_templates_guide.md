# 진단 플롯 템플릿 가이드 (다른 실험/데이터셋에 적용하기)

이 문서는 RedLamp_Check 저장소(`scripts/local_diagnostic_curves.py`)에서 만든 3개의 플롯
템플릿을 요약한다. 다른 세션에서 다른 데이터셋/실험에 이 템플릿을 그대로 적용하거나 참고해서
비슷한 걸 만들 때 쓰라고 정리한 것 — 이 저장소의 실험 히스토리 자체는 다루지 않는다
(그건 `docs/DS3_handoff_context.md` 참고).

모든 함수는 `matplotlib.backends.backend_pdf.PdfPages` 객체(`pdf`)를 첫 인자로 받아서, 그
안에 한 페이지를 `pdf.savefig(fig)`로 추가하는 방식이다. 여러 페이지를 만들려면 그냥
`with PdfPages(out_path) as pdf:` 블록 안에서 함수를 여러 번 부르면 된다.

## 1. Entity 갤러리 페이지 — `plot_entity_gallery_page(pdf, raw, window_size, title='', real_anomaly_spans=None, n_windows=8)`

**용도**: "이 데이터셋에 어떤 entity/시계열들이 있는지" 빠르게 훑어보기. 모델이나 스코어
전혀 필요 없음 — **순수 시각화**, 어떤 데이터셋에도 그대로 재사용 가능.

**레이아웃**: 5행 2열 한 페이지.
- 1행: 2열 전체를 차지, entity의 **전체 시계열** 그대로.
- 2~5행(4행×2열 = 8칸): 그 시계열 안에서 균등한 간격으로 뽑은 **예시 윈도우 8개**
  (실제 모델에 들어갈 윈도우 하나가 어떻게 생겼는지 감을 잡는 용도).

**입력**: `raw`(1D numpy array, 시계열 전체 하나), `window_size`(모델이 실제 쓸 윈도우
길이), `real_anomaly_spans`(선택, `[(start, end), ...]` 형태의 실제/알려진 이상구간 —
있으면 위 아래 패널 모두에 빨간 음영으로 표시, 없으면 그냥 생략).

**예시**: `scripts/build_anomsim_entity_gallery.py`(하이라이트 없음),
`scripts/build_ucr_group_galleries.py`(실제 anomaly 하이라이트 있음).

## 2. 6패널 진단 페이지 — `plot_diagnostic_page(pdf, raw_series, series_list, focus_start, focus_end, window_size, title='', real_anomaly_spans=None)`

**용도**: 모델이 특정 구간을 얼마나 잘/못 재구성하는지 자세히 뜯어보기. **모델의 재구성값과
스코어가 필요함** — 이 저장소의 RedLamp 모델(`main.anomaly_scoreing`/`main.mse`)에서 나온
값을 기준으로 만들었지만, 패널 자체는 "어떤 배열이든 넣으면 그려주는" 순수 플로팅 함수라
**다른 모델의 스코어 배열을 넣어도 그대로 동작**한다.

**6개 패널** (`series_list`의 각 dict가 모델 하나, 1개 또는 여러 개를 겹쳐 그릴 수 있음):
1. raw 신호 + 모델 재구성 (`reconstruction`)
2. `|raw - reconstruction|` (pointwise, 스무딩/정규화 없음)
3. MSE (raw) — 윈도우 전체 평균 재구성오차 (`mse_raw`), 스무딩/정규화 없음
4. MSE_Norm_Smooth score (`mse_score`, [0,1] 정규화, y축 고정)
5. CE_Norm_Smooth score (`ce_score`, [0,1] 정규화, y축 고정) — 분류기가 있는 모델에서만 의미 있음
6. Anomaly_Norm_Smooth score (`score`, [0,1] 정규화, y축 고정) — 선택적으로 `threshold` 값을
   점선으로 표시 가능 (예: TSB_UAD RF의 `mean(score)+3*std(score)`)

**입력**: `raw_series`(1D 시계열, focus_start/focus_end/window_size로 계산되는 전체 구간),
`series_list`는 `dict(label=..., reconstruction=..., mse_score=..., ce_score=..., score=...,
mse_raw=..., threshold=선택)` 리스트. `focus_start=0, focus_end=len(raw_series)`로 부르면
전체 시계열 모드(확대 없이 처음부터 끝까지)가 된다.

**주의**: 분류기(CE score)가 없는 모델이면 `ce_score`에 그냥 0 배열이나 `mse_score`와
동일한 값을 넣어도 되고, 패널 자체를 지우고 싶으면 함수를 복사해서 그 패널만 빼면 된다
(패널 순서/개수가 하드코딩돼 있어서 동적으로 빼는 옵션은 없음).

**예시**: `scripts/build_ucr_test_diagnostics.py`(whole-series, 2모델 오버레이+threshold),
`scripts/build_self_train_val_diagnostics.py`(로컬 윈도우+context 확대 모드).

## 3. 랜덤 윈도우 인스펙터 — `plot_window_inspector_page(pdf, curves_by_model, models_by_label, device, positions, window_size, title='', n_cols=2)`

**용도**: 특정 윈도우 몇 개를 골라서, 그 윈도우 하나만 모델에 다시 통과시켜(배치=1) **진짜
정확한 재구성**을 보고 싶을 때. (템플릿 2의 재구성 곡선은 "각 윈도우의 마지막 timestep만
이어붙인" 근사값이라, 윈도우 중간 부분에서 모델이 실제로 뭘 출력했는지는 안 보여줌 — 이걸
정확히 보려면 이 템플릿이 필요함.)

**입력**: `curves_by_model`(dict label -> 그 모델의 whole-series curves dict, 각 윈도우의
지표 값을 서브플롯 제목에 적는 용도), `models_by_label`(dict label -> 실제 로드된 모델
객체, **`model(window_tensor)` 호출 시 `(predicted, _, _)` 3-tuple을 반환한다고 가정** —
다른 모델 구조를 쓴다면 이 부분만 그 모델의 forward 시그니처에 맞게 고치면 됨), `positions`
(inspect할 timestep 위치 리스트, 각 위치는 "그 위치를 마지막 timestep으로 갖는 윈도우"를
가리킴).

**예시**: `scripts/build_ucr_test_diagnostics.py` (entity당 40개 랜덤 윈도우, 4페이지).

## 재사용 가능한 작은 유틸 함수들 (같은 파일)

- `find_anomaly_segments(labels, max_segments=5)`: 0/1 label 배열에서 연속된 1-구간들을
  `(start, end)` 리스트로 뽑아줌. 아무 label 배열에나 적용 가능.
- `pick_sample_positions(curve_len, window_size, n=5)`: 길이 `curve_len`인 시계열 안에서
  윈도우가 완전히 들어갈 수 있는 위치를 균등 간격으로 `n`개 뽑아줌.
- `window_bounds_from_end_index(end_idx, window_size)`: "이 위치가 윈도우의 마지막
  timestep"이라고 할 때 그 윈도우의 `(start, end)`를 계산.

이 세 개는 모델/데이터셋에 전혀 의존하지 않는 순수 numpy 로직이라 그대로 복사해서 다른
프로젝트에 써도 된다.

## 요약: 새 데이터셋에 뭐부터 쓰면 되나

- **모델 없이, 그냥 데이터 구경만 하고 싶다** → 템플릿 1(`plot_entity_gallery_page`)만
  있으면 충분. raw 시계열 배열만 있으면 됨.
- **모델의 재구성/이상탐지 스코어까지 보고 싶다** → 템플릿 2. 단, 그 모델의 스코어링 방식이
  이 저장소의 `main.anomaly_scoreing`과 다르면, `series_list`에 넣을 `mse_score`/`ce_score`/
  `score`를 그 실험 자체의 스코어링 함수로 직접 계산해서 넣어야 함 (템플릿 자체는 값이
  어디서 왔는지 모름, 그냥 그려줄 뿐).
- **왜 특정 시점에서 두 모델의 평가가 다르게 나오는지 파고들고 싶다** → 템플릿 3.
