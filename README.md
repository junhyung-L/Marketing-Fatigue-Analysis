# marketing-mix
## 추가 피처(Features) — 무엇 / 왜 / 어떻게 / 도메인지식

## 목차
1. [선행연구 1) RFM / BTYD (재구매 타이밍·휴지기)](#선행연구-1-rfm--btyd-재구매-타이밍휴지기)  
2. [선행연구 2) Temporal Dynamics (시간/요일/달력)](#선행연구-2-temporal-dynamics-시간요일달력)  
3. [선행연구 3) Behavior & Campaign Performance (빈도·리듬·최근행동·컨텍스트)](#선행연구-3-behavior--campaign-performance-빈도리듬최근행동컨텍스트)  
4. [선행연구 4) Novelty / Journey (신선도·경로 정합성)](#선행연구-4-novelty--journey-신선도경로-정합성)  
5. [New) 본 연구 제안 변수군 (HTE·ECC·전달성·슬롯컨디션)](#new-본-연구-제안-변수군-hteecc전달성슬롯컨디션)  
6. [공통 계산 원칙](#공통-계산-원칙)  
7. [한눈 요약(매핑)](#한눈-요약매핑)

---

## 선행연구 1) RFM / BTYD (재구매 타이밍·휴지기)

**핵심:** Recency, Frequency, Monetary / Interpurchase time

- `days_since_last_purchase`  
  - **무엇:** 최근 구매 후 경과일(Recency)  
  - **왜:** 최근성↑ → 재구매·마케팅 반응성↑  
  - **어떻게:** `now - last_buy_time` 일수

- `feat_rtb_hazard`  
  - **무엇:** 개인 구매 간격(μ, σ)로 “지금이 때인가” 근사  
  - **왜:** 카테고리별 반복 주기 반영  
  - **어떻게:** 간격 통계(부족 시 글로벌 μ/σ) → 가우시안형 준비도

- `feat_postbuy_refrac`  
  - **무엇:** 구매 직후 불응기(Refractory) 감쇠  
  - **왜:** 포만·예산 전환으로 즉시 재구매↓  
  - **어떻게:** `exp(-Δt / tau)` 형태 감쇠

---

## 선행연구 2) Temporal Dynamics (시간/요일/달력)

**핵심:** 개인 리듬 정렬·달력 효과(급여/월말/분기말)

- `feat_hour_shift`, `feat_dow_shift`  
  - **무엇:** 과거 **반응** 선호각과 현재 발송각의 원형 거리 `2·sin(Δ/2)`  
  - **왜:** 개인 골든타임 정렬 시 반응률↑  
  - **어떻게:** 반응 각도 누적 → `arctan2`로 선호각 추정 → 현재각과 차이

- 달력 신호  
  - `cal_is_weekend`, `cal_week_of_month`  
  - `feat_payday_bump`, `feat_monthend_bump`, `feat_eoq_bump`  
  - **무엇:** 주말/월 주차/급여·월말·분기말 근접도  
  - **왜:** 시즌성·현금흐름 주기 반영  
  - **어떻게:** 현지(Asia/Seoul) 캘린더 파생 + Gaussian bump

---

## 선행연구 3) Behavior & Campaign Performance (빈도·리듬·최근행동·컨텍스트)

### A. 빈도·리듬(피로/쿨다운/간격 변동성)

- `feat_fatigue`  
  - **무엇:** 채널별 발송 잔상의 지수감쇠 합  
  - **왜:** 과발송 → 관심·전달성↓  
  - **어떻게:** `∑ exp(-(t - t_i)/tau_channel(i))` (현재행 제외)

- `feat_cooldown_ok`  
  - **무엇:** 동일 채널 최소 간격 충족 플래그  
  - **왜:** 평판·반응 악화 방지  
  - **어떻게:** 유저×채널 `diff >= threshold`

- `feat_last_<ch>_hours`, `feat_last_any_hours`  
  - **무엇:** 마지막 접촉 이후 경과시간  
  - **왜:** 너무 잦아도/너무 뜸해도 성과 저해  
  - **어떻게:** 시간 정렬 `shift(1)` 차이(초기 결측은 글로벌 중앙값)

- `u_cadence_std_30d`  
  - **무엇:** 최근 30일 발송 **간격** 표준편차  
  - **왜:** 리듬 불규칙성↑ → 기대 형성↓  
  - **어떻게:** 롤링 “30D”(현재행 제외, 데이터 적으면 0)

### B. 최근 개인 행동(7/30일)

- `u_open_cnt_7d/30d`, `u_click_cnt_7d/30d`, `u_buy_cnt_7d/30d`  
- `u_open_rate_7d/30d`, `u_click_rate_7d/30d`, `u_buy_rate_7d/30d`  
  - **무엇:** 단기 관여도 지표  
  - **왜:** 강한 예측자  
  - **어떻게:** 롤링 집계(현재행 제외), 분모 최소 1.0

### C. 동시대 컨텍스트 성과(집단 신호)

- `(토픽×채널) ctx_tc_open_rate_7d/30d, ctx_tc_buy_rate_7d/30d`  
- `(캠페인ID) ctx_camp_open_rate_7d/30d, ctx_camp_buy_rate_7d/30d`  
  - **무엇:** 맥락 “온도”  
  - **왜:** 시장/플랫폼/크리에이티브 변화 반영  
  - **어떻게:** 키별 롤링 비율(희소 시 클리핑)

---

## 선행연구 4) Novelty / Journey (신선도·경로 정합성)

- `feat_topic_novelty` (보조: `topic_N7`, `topic_t_since_hours`)  
  - **무엇:** 과노출-회복의 동시 반영  
  - **어떻게:** `exp(-N7/κ_ch) * (1 - exp(-Δt/τ_ch))`

- `feat_like_last_success`  
  - **무엇:** 마지막 **구매 유발** 캠페인과 현재 속성의 자카드 유사도  
  - **왜:** “성공 레시피” 재현

- `feat_path_align`  
  - **무엇:** 구매 직전 L-길이 채널 시퀀스 프로토타입 vs 최근 시퀀스의 bigrams 자카드  
  - **왜:** 채널 **순서**가 여정에 영향

---

## New) 본 연구 제안 변수군 (HTE·ECC·전달성·슬롯컨디션)

### A. 개인-변형 HTE (A/B·토글 민감도)

- `feat_ab_sens`, `feat_ab_unc`, `feat_ab_mask` *(출처키: `src_abtest`, `src_toggle`, `src_campaign`)*  
  - **무엇:** 개인×컨텍스트×변형 효과의 확률 가중 결합/불확실도  
  - **어떻게:** `w = n/(n+τ)` 가중, 최소 표본 미달 `mask=0`

- `feat_tg_<key>_sens`, `feat_tg_<key>_unc`  
  - **무엇:** 카피 토글(예: deadline/emoji/discount 등)별 개인 민감도·불확실도  
  - **어떻게:** 개인 누적 Beta(1,1) 사전 기반 비율/분산

### B. ECC(취향-크리에이티브 정합성)

- `feat_ecc_hamming`  
  - **무엇:** 개인 토글 선호 프로필 vs 현재 토글 조합 해밍거리 평균  
  - **왜:** 불일치↑ → 설득 비용↑

### C. 슬롯 마이크로클라이밋(단기 컨텍스트 건강)

- `feat_microclimate_z`  
  - **무엇:** (도메인×요일×2시간) **8주 롤링 오픈율**이 베이스 대비 얼마나 벗어났는지 z-score  
  - **왜:** 단기 슬롯 컨디션 탐지(SPC 개념)

### D. 전달성/프로바이더 & 개인 리스크

- 프로바이더 건강: `prov_open7/30/delta`, `prov_bounce7/30/delta`, `prov_comp7/30/delta`  
  - **무엇:** 도메인 평판의 단·장기 비율/델타  
  - **왜:** 인박스 진입률에 직접 영향  
  - **어떻게:** 롤링 비율(현재행 제외, `[ε,1−ε]` 클리핑)

- 개인 전달성 EWMA: `feat_user_deliv_bnc`, `feat_user_deliv_cmp`, `feat_user_deliv_any`  
  - **무엇:** 바운스/불만의 시간감쇠 누적 점수  
  - **왜:** 고위험 수신자 반복 접촉 회피

---

## 공통 계산 원칙

- **타임존 일관:** 모든 시계열 `UTC → Asia/Seoul` 변환 후 파생  
- **현재행 누출 방지:** 롤링·집계는 **현재행 제외**(예: `closed="left"`)  
- **희소/초기값 방어:** 중앙값/0/클리핑·half-life로 안정화  
- **표준화:** 비율은 `[ε, 1−ε]` 클리핑, z-score는 적절한 클립(±5)  
- **키 관리:** 개인키×컨텍스트키(토픽×채널/캠페인ID/슬롯키)를 명시적으로 분리

---

## 한눈 요약(매핑)

- **선행연구 1 (RFM/BTYD):** `days_since_last_purchase`, `feat_rtb_hazard`, `feat_postbuy_refrac`  
- **선행연구 2 (Temporal):** `feat_hour_shift`, `feat_dow_shift`, `cal_is_weekend`, `cal_week_of_month`, `feat_payday_bump`, `feat_monthend_bump`, `feat_eoq_bump`  
- **선행연구 3 (Behavior/Performance):** `feat_fatigue`, `feat_cooldown_ok`, `feat_last_*_hours`, `u_cadence_std_30d`, `u_*_cnt/rate_7d/30d`, `ctx_*_rate_7d/30d`  
- **선행연구 4 (Novelty/Path):** `feat_topic_novelty`, `topic_N7`, `topic_t_since_hours`, `feat_like_last_success`, `feat_path_align`  
- **New (본 연구 제안):** `feat_ab_*`, `feat_tg_*_sens/unc`, `feat_ecc_hamming`, `feat_microclimate_z`, `feat_user_deliv_*`, `prov_*_delta`
