# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from umap import UMAP
# import plotly.express as px
# from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
from umap import UMAP
import plotly.express as px
from sentence_transformers import SentenceTransformer
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go


def render_right_sidebar():
    # 카테고리별 단어 리스트
    # DFS 관련 단어 리스트
    dfs_words = [
        "깊이우선탐색", "트리탐색", "노드추적", "재귀호출", "스택자료구조", 
        "그래프순회", "백트래킹", "자식노드", "루트노드", "스택기반탐색", 
        "노드방문", "그래프탐색", "순환호출", "트리구조", "부모노드", 
        "스택활용", "깊이탐색", "순회경로", "재귀적탐색", "스택추적"
    ]

    # BFS 관련 단어 리스트
    bfs_words = [
        "너비우선탐색", "큐자료구조", "레벨탐색", "최단경로탐색", "그래프탐험", 
        "연결탐색", "방문기록", "인접노드탐색", "큐구현", "탐색순서", 
        "큐기반탐색", "그래프순환", "레벨별방문", "인접리스트", "그래프큐", 
        "큐탐색", "너비탐색구현", "큐노드", "탐색노드", "연결구조"
    ]

    # Sort Algorithm 관련 단어 리스트
    sort_words = [
        "정렬알고리즘", "버블정렬", "선택정렬", "삽입정렬", "합병정렬",
        "퀵정렬", "힙정렬", "기수정렬", "배열정렬", "리스트정렬",
        "정렬비교", "정렬구조", "정렬순서", "데이터정렬", "정렬속도",
        "정렬방식", "정렬조건", "정렬효율성", "정렬구현", "정렬시간복잡도"
    ]

    # Greedy Algorithm 관련 단어 리스트
    greedy_words = [
        "탐욕적선택", "최적해구성", "최적화알고리즘", "탐욕전략", "탐욕적탐색",
        "탐욕적해결법", "최대가치선택", "비용최소화", "탐욕적분석", "탐욕적패턴",
        "탐욕적구조", "탐욕적단계", "탐욕적연산", "탐욕적결정", "탐욕적최적화",
        "탐욕적구현", "탐욕적방법", "탐욕적문제", "탐욕적효율성", "탐욕적구성방법"
    ]

    # DP 관련 단어 리스트
    dp_words = [
        "동적계획법", "부분문제", "최적부분구조", "메모이제이션", "캐시활용",
        "점화식", "최적구조", "DP알고리즘", "DP구현", "하향식",
        "상향식", "중복계산제거", "최적해구조", "DP패턴", "부분최적화",
        "DP효율성", "DP활용", "DP시간복잡도", "DP구조", "최적화분할"
    ]

    # Shortest Path 관련 단어 리스트
    shortest_path_words = [
        "최단거리", "다익스트라", "벨만포드", "플로이드와샬", "경로탐색",
        "그래프가중치", "최소경로", "최단경로탐색", "거리계산", "경로효율성",
        "최단경로구현", "그래프구조", "최소비용", "경로연산", "최단경로분석",
        "최적경로탐색", "그래프경로", "최단경로패턴", "최단경로시간", "최단경로설계"
    ]

    # model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    # # 모든 단어와 카테고리 데이터프레임 생성
    # categories = ['DFS'] * len(dfs_words) + ['BFS'] * len(bfs_words) + ['Sort'] * len(sort_words) + \
    #              ['Greedy'] * len(greedy_words) + ['DP'] * len(dp_words) + ['Shortest Path'] * len(shortest_path_words)

    # words = dfs_words + bfs_words + sort_words + greedy_words + dp_words + shortest_path_words
    # df = pd.DataFrame({'Word': words, 'Category': categories})

    # # 텍스트 임베딩 (Sentence Transformer)
    # embeddings = model.encode(df['Word'].tolist(), show_progress_bar=True)
    
    # # 차원 축소 (UMAP)
    # umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    # embedding = umap_model.fit_transform(embeddings)

    # # 결과를 데이터프레임에 저장
    # df['UMAP_1'] = embedding[:, 0]
    # df['UMAP_2'] = embedding[:, 1]

    # # Streamlit 사이드바 및 UI
    # st.header("Algorithm Word Categories")
    # selected_categories = st.multiselect(
    #     "Select categories to display:",
    #     options=df['Category'].unique(),
    #     default=df['Category'].unique()
    # )

    # # 선택한 카테고리 필터링
    # filtered_df = df[df['Category'].isin(selected_categories)]

    # # Plotly 시각화
    # fig = px.scatter(
    #     filtered_df,
    #     x='UMAP_1',
    #     y='UMAP_2',
    #     color='Category',
    #     text='Word',
    #     # title="Algorithm Word Embedding Visualization",
    #     labels={'UMAP_1': 'Dimension 1', 'UMAP_2': 'Dimension 2'},
    #     hover_data=['Word']
    # )

    # st.plotly_chart(fig, use_container_width=True)
    # Sentence Transformer 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    # 모든 단어와 카테고리 데이터프레임 생성
    categories = ['DFS'] * len(dfs_words) + ['BFS'] * len(bfs_words) + ['Sort'] * len(sort_words) + \
                 ['Greedy'] * len(greedy_words) + ['DP'] * len(dp_words) + ['Shortest Path'] * len(shortest_path_words)

    words = dfs_words + bfs_words + sort_words + greedy_words + dp_words + shortest_path_words
    df = pd.DataFrame({'Word': words, 'Category': categories})

    # 텍스트 임베딩 (Sentence Transformer)
    embeddings = model.encode(df['Word'].tolist(), show_progress_bar=True)

    # 차원 축소 (UMAP)
    umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding = umap_model.fit_transform(embeddings)

    # 결과를 데이터프레임에 저장
    df['UMAP_1'] = embedding[:, 0]
    df['UMAP_2'] = embedding[:, 1]

    # Streamlit 사이드바 및 UI
    st.header("Algorithm Word Categories")
    selected_categories = st.multiselect(
        "Select categories to display:",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )

    # 선택한 카테고리 필터링
    filtered_df = df[df['Category'].isin(selected_categories)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['UMAP_1'],
        y=df['UMAP_2'],
        mode='markers',
        hovertemplate='<b>%{text}</b><extra></extra>',  # Show only the word in hover text
    marker=dict(size=10, color='blue')
    
    ))

    fig.update_layout(
    title='Word Embeddings Visualization',
    xaxis_title='X',
    yaxis_title='Y',
    height=500,
    clickmode='event+select',
    legend=dict(
            orientation="h",  # 수평으로 정렬
            yanchor="bottom",
            y=-0.2,  # 그래프 아래로 이동
            xanchor="center",
            x=0.5
        )
    )

    # 그래프 축 숨기기
    fig.update_xaxes(visible=False)  # x축 숨김
    fig.update_yaxes(visible=False)  # y축 숨김

    # # Plotly 시각화
    # fig = px.scatter(
    #     filtered_df,
    #     x='UMAP_1',
    #     y='UMAP_2',
    #     color='Category',
    #     text='Word',
    #     labels={'UMAP_1': 'Dimension 1', 'UMAP_2': 'Dimension 2'},
    #     hover_data=['Word']  # Hover 시 단어 표시
    # )

    # streamlit_plotly_events로 클릭 이벤트 감지
    clicked_points = plotly_events(fig, click_event=True, hover_event=False)

    # 클릭된 점의 정보 출력
    if clicked_points:
        clicked_index = clicked_points[0]['pointIndex']  # 클릭된 점의 인덱스 가져오기
        clicked_word = filtered_df.iloc[clicked_index]['Word']  # 클릭된 점의 단어 가져오기
        st.write(f"### You clicked on the word: **{clicked_word}**")