# 아키드로우 사용자 추천시스템

## 파일설명
  - `data_preprocess`: 데이터 전처리 코드
  - `contents-based.ipynb`: 콘텐츠 기반 필터링을 사용하여 cold-start 문제를 해결하기 위한 초기 유저에 대한 관련 아이템 추천시스템
  - `CF-SGD`: SGD 알고리즘을 활용한 잠재요인 협업 필터링을 기반으로 어느정도 유저에 대한 정보가 들어올 경우 콘텐츠 기반 필터링과 함께 하이브리드 추천시스템 대안
  - `CF-KNN`: Item-based collaborative filtering ⇒ 특정 사용자가준 점수간의 유사한 상품을 찾아 추천 (하이브리드 추천시스템 대안)
  - `item2vec`: cold-start 해결을 위해서 전문가들이 선택한 아이템 셋을 학습하여 word2vec를 활용한 콘텐츠 기반 필터링

## 업데이트 예정
  - 미정
