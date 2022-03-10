# 아키드로우 사용자 추천시스템

## Contributor
  - 유치영
  - 이루오

## 파일설명
  - `data_preprocess.py` : 데이터 전처리 코드
  - `text-distance.ipynb` (최종 선택 모델) : 콘텐츠 기반 필터링 (아이템 정보 기반 거리 유사도를 측정하기 때문에 검색 속도가 빠름)
  
  베이스 모델들
  - `contents-based.ipynb` : count vectorizer 콘텐츠 기반 필터링을 사용하여 cold-start 문제를 해결하기 위한 초기 유저에 대응
  - `CF-SGD.ipynb` : SGD 알고리즘을 활용한 잠재요인 협업 필터링을 기반으로 어느정도 유저에 대한 정보가 들어올 경우 콘텐츠 기반 필터링과 함께 하이브리드 추천시스템 대안
  - `CF-KNN.ipynb` : Item-based collaborative filtering ⇒ 특정 사용자가준 점수간의 유사한 상품을 찾아 추천 (하이브리드 추천시스템 대안)
  - `item2vec.ipynb` : cold-start 해결을 위한 embedding 알고리즘의 word2vec를 활용한 콘텐츠 기반 필터링
  - `hybrid.ipynb` : 콘텐츠 기반 필터링 + 잠재요인 협업 필터링 하이브리드 추천시스템 
  

## 업데이트
  - `2022-03-07`
    - 콘텐츠기반 필터링, 협업필터링 SGD, 협업필터링 KNN, item2vec 모델파일 추가
    - 추가 데이터 적용
  - `2022-03-08`
    - 단일, 복수 아이템에 대한 하이브리드 추천시스템 모델 추가
  - `2022-03-10`
    - 텍스트 거리 유사도를 기반으로 한 콘텐츠 기반 필터링 모델 추가 (속도 개선 : 기존 20초~2분 -> 0.8~4초 / 1개 아이템에 대한 추천 검색)

## 다음 주 업데이트 예정
  - `top-style` 에 따라 아이템 추천 검색 성능 향상
  - 복수 아이템에 대한 추천 검색 알고리즘 개선 및 성능 향상
  - 추후 `room type` 등 데이터에 새로운 feature가 추가 된 경우 cold-start를 해소하기 위한 방향으로 적용 가능 할 수 있도록 알고리즘 개선
  - 완성된 추천 시스템의 성능 개선을 정량적으로 확인할 수 있는 객관적 평가 지표 마련
