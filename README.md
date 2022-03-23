# 아키드로우 사용자 추천시스템
## Environment
  - MacOS, Windows
  - Python 3.8
  - Jupyter Notebook
  - GitHub

## Contributor
  - 유치영
    - 데이터를 추천/검색 알고리즘 성능 개선을 위한 다양한 전처리
    - 전처리 코드 클래스화를 통해 새 데이터 통합 대응 등 end2end 모델을 위한 자동화
    - 추천/검색 알고리즘 평가지표 개발
  - 이루오
    - 콘텐츠 기반 추천/검색 모델 탐색 및 선정 후 클래스화
    - 데이터 입력부터 추천결과까지 end2end 자동화 모델을 위해 데이터클래스 + 모델클래스 파일 취합
    - 고객 맞춤형 추천을 위해 모델 최적화 및 검색 필터링 정확도, 속도 개선

## 사용법
  ```
  conda create -n recommend python=3.8
  conda activate recommend
  cd Recommendation_system
  python -m pip install -r requirements.txt
  python run.py
  ```
  - 추천 결과 목록과 추천된 상품 이미지는 동일 디렉터리에 `result_{검색아이템명}/` 디렉터리를 자동생성 후 하위에 `.png`,`.csv` 파일로 저장
  ### Demonstration
  ![test_demo](https://user-images.githubusercontent.com/61719257/158094840-1e0bfa06-82e5-4243-8416-9e22d34945f5.gif)


## 파일설명
  - `preprocess.py` : 데이터 전처리 클래스
  - `run.py` (실행 파일) : 콘텐츠 기반 추천 검색 필터링
  - `test-result.ipynb` : 모델 실행 추천 결과 보고서 (쥬피터 파일)
  - `evaluation.ipynb` : 모델 평가 보고서 (쥬피터 파일)
  - `algorithm` 디렉터리 내 모델들
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
    - 텍스트 거리 유사도를 기반으로 한 콘텐츠 기반 필터링 모델 추가 (속도 개선 : 기존 20초-2분 -> 약 0.2초 / 1개 아이템에 대한 추천 검색)
  - `2022-03-12`
    - 아이템 추천 시 아이템 카테고리보다 아이템 특징에 기반하여 추천하는 방향으로 알고리즘 개선 + 새로운 카테고리 태그를 통해 정교한 필터링을 위한 데이터 전처리
    - `top-style` 에 따라 아이템 추천 검색 성능 향상
    - 복수 아이템에 대한 추천 검색 알고리즘 개선 및 성능 향상
    - factorization + 사용 가이드 (수정 중 - README 파일에 추가 예정)
  - `2022-03-13`
    - 완성된 추천 시스템의 성능 개선을 정량적으로 확인할 수 있는 객관적 평가 지표 마련
  - `2022-03-14`
    - 최종 데이터 => 모델 적용 및 테스트
  - `2022-03-17`
    - `top-style` 에 따른 가중치 적용
    - 아이템 검색 결과 개선 (precision : 0.2-0.3 => 0.79-0.82)
    - 평가 지표 속도에 대한 개선
  - `2022-03-18`
    - '소파, Sofas' 카테고리에 대해서 필터링 고도화를 통해 모델 성능 개선
       
       ![Mar-21-2022 11-57-41](https://user-images.githubusercontent.com/61719257/159199665-b63862a5-9ed4-48d9-8dcd-56d20d463707.gif)

    - 불필요한 카테고리 제거
      ```
      a. 421의경우 제거 대상
        - Construction
        - Appliances
        - Bathroom
        - Kitchen
        - Outdoor
      b. B57 의 경우 제거 대상
        - 문/창문
        - 가전
        - 주방싱크/욕실
        - 파티션/구조물
      ```
  - `2022-03-21`
    - 학습 데이터 유효성 확인을 위해 불필요한 카테거리 제거 후 같은 프로젝트 내 아이템이 1~2개만 있는 경우 제거 
  - `2022-03-23`
    - product_id 는 동일하나 색상이 다르고 이름은 같아서 헷갈릴 수 있는 아이템에 대해 추천 결과 display 시 색상 정보가 있는 아이템은 색상정보가 함께 표기되어 사용자에게 줄 수 있는 혼란을 방지하도록 업데이트
    
## 업데이트 예정
  - 미정
  
## 가구 개인화 추천시스템 향후 발전 방향
  - 하이브리드 추천 시스템 발전 가능성
    - 사용자들의 정보가 충분히 쌓이면 [사용자-아이템] 테이블 정보를 통해 사용자간 유사도를 계산하여 비슷한 취향의 사용자를 찾아주고 그에 해당하는 아이템들을 추천해주는 방향
    
    ![future_plan](https://user-images.githubusercontent.com/61719257/159622620-075af858-32a7-49f7-bcb0-9b035094c9e3.gif)
