# # import streamlit as st
# # import pandas as pd
# # from umap import UMAP
# # import plotly.express as px
# # from sentence_transformers import SentenceTransformer
# # from streamlit_plotly_events import plotly_events
# # import plotly.graph_objects as go
# # from plotly.graph_objs import Scatter, Figure, Layout  


# # def render_right_sidebar():
# #     # 카테고리별 단어 리스트
# #     # DFS 관련 단어 리스트
# #     dfs_words = [
# #         "깊이우선탐색", "트리탐색", "노드추적", "재귀호출", "스택자료구조", 
# #         "그래프순회", "백트래킹", "자식노드", "루트노드", "스택기반탐색", 
# #         "노드방문", "그래프탐색", "순환호출", "트리구조", "부모노드", 
# #         "스택활용", "깊이탐색", "순회경로", "재귀적탐색", "스택추적"
# #     ]

# #     # BFS 관련 단어 리스트
# #     bfs_words = [
# #         "너비우선탐색", "큐자료구조", "레벨탐색", "최단경로탐색", "그래프탐험", 
# #         "연결탐색", "방문기록", "인접노드탐색", "큐구현", "탐색순서", 
# #         "큐기반탐색", "그래프순환", "레벨별방문", "인접리스트", "그래프큐", 
# #         "큐탐색", "너비탐색구현", "큐노드", "탐색노드", "연결구조"
# #     ]

# #     # Sort Algorithm 관련 단어 리스트
# #     sort_words = [
# #         "정렬알고리즘", "버블정렬", "선택정렬", "삽입정렬", "합병정렬",
# #         "퀵정렬", "힙정렬", "기수정렬", "배열정렬", "리스트정렬",
# #         "정렬비교", "정렬구조", "정렬순서", "데이터정렬", "정렬속도",
# #         "정렬방식", "정렬조건", "정렬효율성", "정렬구현", "정렬시간복잡도"
# #     ]

# #     # Greedy Algorithm 관련 단어 리스트
# #     greedy_words = [
# #         "탐욕적선택", "최적해구성", "최적화알고리즘", "탐욕전략", "탐욕적탐색",
# #         "탐욕적해결법", "최대가치선택", "비용최소화", "탐욕적분석", "탐욕적패턴",
# #         "탐욕적구조", "탐욕적단계", "탐욕적연산", "탐욕적결정", "탐욕적최적화",
# #         "탐욕적구현", "탐욕적방법", "탐욕적문제", "탐욕적효율성", "탐욕적구성방법"
# #     ]

# #     # DP 관련 단어 리스트
# #     dp_words = [
# #         "동적계획법", "부분문제", "최적부분구조", "메모이제이션", "캐시활용",
# #         "점화식", "최적구조", "DP알고리즘", "DP구현", "하향식",
# #         "상향식", "중복계산제거", "최적해구조", "DP패턴", "부분최적화",
# #         "DP효율성", "DP활용", "DP시간복잡도", "DP구조", "최적화분할"
# #     ]

# #     # Shortest Path 관련 단어 리스트
# #     shortest_path_words = [
# #         "최단거리", "다익스트라", "벨만포드", "플로이드와샬", "경로탐색",
# #         "그래프가중치", "최소경로", "최단경로탐색", "거리계산", "경로효율성",
# #         "최단경로구현", "그래프구조", "최소비용", "경로연산", "최단경로분석",
# #         "최적경로탐색", "그래프경로", "최단경로패턴", "최단경로시간", "최단경로설계"
# #     ]

# #     # Sentence Transformer 모델 로드
# #     model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# #     # 모든 단어와 카테고리 데이터프레임 생성
# #     categories = ['DFS'] * len(dfs_words) + ['BFS'] * len(bfs_words) + ['Sort'] * len(sort_words) + \
# #                  ['Greedy'] * len(greedy_words) + ['DP'] * len(dp_words) + ['Shortest Path'] * len(shortest_path_words)
# #     words = dfs_words + bfs_words + sort_words + greedy_words + dp_words + shortest_path_words
# #     df = pd.DataFrame({'Word': words, 'Category': categories})

    
# #     # 텍스트 임베딩 (Sentence Transformer)
# #     embeddings = model.encode(df['Word'].tolist(), show_progress_bar=True)

# #     # 차원 축소 (UMAP)
# #     umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
# #     embedding = umap_model.fit_transform(embeddings)

# #     # 결과를 데이터프레임에 저장
# #     df['UMAP_1'] = embedding[:, 0]
# #     df['UMAP_2'] = embedding[:, 1]

# #     # Streamlit 사이드바 및 UI
# #     st.header("Algorithm Word Categories")
# #     selected_categories = st.multiselect(
# #         "Select categories to display:",
# #         options=df['Category'].unique(),
# #         default=df['Category'].unique()
# #     )

# #     # 선택한 카테고리 필터링
# #     filtered_df = df[df['Category'].isin(selected_categories)]

# #     # Plotly 시각화를 위한 데이터 생성
# #     fig = Figure()

# #     # 카테고리별로 데이터 추가
# #     category_colors = {
# #         "DFS": "blue",
# #         "BFS": "red",
# #         "Sort": "green",
# #         "Greedy": "purple",
# #         "DP": "orange",
# #         "Shortest Path": "brown"
# #     }

# #     for category in filtered_df['Category'].unique():
# #         category_df = filtered_df[filtered_df['Category'] == category]
# #         fig.add_trace(Scatter(
# #             x=category_df['UMAP_1'],
# #             y=category_df['UMAP_2'],
# #             mode='markers',
# #             marker=dict(size=8, color=category_colors[category]), opacity=0.5,
# #             name=category,  # 범례에 표시
# #             text=category_df['Word'],  # Hover 시 단어 표시
# #             hovertemplate='<b>%{text}</b><extra></extra>'
# #         ))

# #     # 그래프 레이아웃 설정
# #     fig.update_layout(
# #         title=None,  # 제목 제거
# #         xaxis_title=None,
# #         yaxis_title=None,
# #         height=450,
# #         width=500,
# #         clickmode='event+select',
# #         legend=dict(
# #             orientation="h",  # 수평으로 정렬
# #             yanchor="bottom",
# #             y=-0.3,  # 그래프 아래로 이동
# #             xanchor="center",
# #             x=0.5
# #         )
# #     )

# #     # 그래프 축 숨기기
# #     fig.update_xaxes(visible=False)
# #     fig.update_yaxes(visible=False)
# #     # Streamlit CSS 스타일로 왼쪽 정렬
# #     st.markdown(
# #         """
# #         <style>
# #         .plot-container {
# #             display: flex;
# #             justify-content: flex-start; /* 왼쪽 정렬 */
# #             margin-left: 10px;
# #         }
# #         </style>
# #         """,
# #         unsafe_allow_html=True
# #     )

# #     # streamlit_plotly_events로 클릭 이벤트 감지
# #     # clicked_points = plotly_events(fig, click_event=True, hover_event=False)

# #     # Plotly 그래프 표시 (왼쪽 정렬)
# #     st.markdown('<div class="plot-container">', unsafe_allow_html=True)
# #     # st.plotly_chart(fig, use_container_width=False)
# #     st.markdown('</div>', unsafe_allow_html=True)

# #     # streamlit_plotly_events로 클릭 이벤트 감지
# #     clicked_points = plotly_events(fig, click_event=True, hover_event=False)

# #     # Plotly 그래프 표시
# #     # st.plotly_chart(fig, use_container_width=True)
import streamlit as st
import pandas as pd
from umap import UMAP
from sentence_transformers import SentenceTransformer
from streamlit_plotly_events import plotly_events
from plotly.graph_objs import Scatter, Figure
# from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.colors import qualitative


# 텍스트 임베딩 함수 (캐싱 적용)
@st.cache_data
def get_embeddings(words):
    # model = SentenceTransformer('jhgan/ko-sroberta-multitask') #한국어 임베딩 모델
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2') #영어 임베딩 모델
    return model.encode(words, show_progress_bar=True)

# UMAP 차원 축소 함수 (캐싱 적용)
@st.cache_data
def apply_umap(embeddings):
    umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    return umap_model.fit_transform(embeddings)

def render_right_sidebar():
    # 카테고리별 단어 리스트
    categories = {
        "DFS": [
            # "깊이우선탐색", "트리탐색", "노드추적", "재귀호출", "스택자료구조", 
            # "그래프순회", "백트래킹", "자식노드", "루트노드", "스택기반탐색", 
            # "노드방문", "그래프탐색", "순환호출", "트리구조", "부모노드", 
            # "스택활용", "깊이탐색", "순회경로", "재귀적탐색", "스택추적"
            "Depth-First Search", "Tree Traversal", "Node Tracking", "Recursive Call", "Stack Data Structure",
            "Graph Traversal", "Backtracking", "Child Node", "Root Node", "Stack-Based Search",
            "Node Visit", "Graph Search", "Recursive Call", "Tree Structure", "Parent Node",
            "Stack Utilization", "Deep Search", "Traversal Path", "Recursive Search", "Stack Trace"
        ],
        "BFS": [
            # "너비우선탐색", "큐자료구조", "레벨탐색", "최단경로탐색", "그래프탐험", 
            # "연결탐색", "방문기록", "인접노드탐색", "큐구현", "탐색순서", 
            # "큐기반탐색", "그래프순환", "레벨별방문", "인접리스트", "그래프큐", 
            # "큐탐색", "너비탐색구현", "큐노드", "탐색노드", "연결구조"
            "Breadth-First Search", "Queue Data Structure", "Level Search", "Shortest Path Search", "Graph Exploration", 
            "Connection Search", "Visited Records", "Adjacent Node Search", "Queue Implementation", "Search Order", 
            "Queue-Based Search", "Graph Traversal", "Level-Wise Visit", "Adjacency List", "Graph Queue", 
            "Queue Search", "Breadth-First Search Implementation", "Queue Node", "Search Node", "Connection Structure"
        ],
        "Sort": [
            # "정렬알고리즘", "버블정렬", "선택정렬", "삽입정렬", "합병정렬",
            # "퀵정렬", "힙정렬", "기수정렬", "배열정렬", "리스트정렬",
            # "정렬비교", "정렬구조", "정렬순서", "데이터정렬", "정렬속도",
            # "정렬방식", "정렬조건", "정렬효율성", "정렬구현", "정렬시간복잡도"
            "Sorting Algorithm", "Bubble Sort", "Selection Sort", "Insertion Sort", "Merge Sort", 
            "Quick Sort", "Heap Sort", "Radix Sort", "Array Sorting", "List Sorting", 
            "Sort Comparison", "Sort Structure", "Sort Order", "Data Sorting", "Sorting Speed", 
            "Sorting Method", "Sorting Conditions", "Sorting Efficiency", "Sort Implementation", "Sorting Time Complexity"
        ],
        "Greedy": [
            "Greedy Selection", "Optimal Solution Construction", "Optimization Algorithm", "Greedy Strategy", "Greedy Search", 
            "Greedy Solution Method", "Maximum Value Selection", "Cost Minimization", "Greedy Analysis", "Greedy Pattern", 
            "Greedy Structure", "Greedy Step", "Greedy Operation", "Greedy Decision", "Greedy Optimization", 
            "Greedy Implementation", "Greedy Approach", "Greedy Problem", "Greedy Efficiency", "Greedy Construction Method"
        ],
        "DP": [
            "Dynamic Programming", "Subproblems", "Optimal Substructure", "Memoization", "Cache Utilization", 
            "Recurrence Relation", "Optimal Structure", "DP Algorithm", "DP Implementation", "Top-Down", 
            "Bottom-Up", "Eliminating Redundant Calculations", "Optimal Solution Structure", "DP Pattern", "Partial Optimization", 
            "DP Efficiency", "DP Application", "DP Time Complexity", "DP Structure", "Optimization Partitioning"
        ],
        "Shortest Path": [
            # "최단거리", "다익스트라", "벨만포드", "플로이드와샬", "경로탐색",
            # "그래프가중치", "최소경로", "최단경로탐색", "거리계산", "경로효율성",
            # "최단경로구현", "그래프구조", "최소비용", "경로연산", "최단경로분석",
            # "최적경로탐색", "그래프경로", "최단경로패턴", "최단경로시간", "최단경로설계"
            "Shortest Distance", "Dijkstra", "Bellman-Ford", "Floyd-Warshall", "Path Search", 
            "Graph Weights", "Minimum Path", "Shortest Path Search", "Distance Calculation", "Path Efficiency", 
            "Shortest Path Implementation", "Graph Structure", "Minimum Cost", "Path Operation", "Shortest Path Analysis", 
            "Optimal Path Search", "Graph Path", "Shortest Path Pattern", "Shortest Path Time", "Shortest Path Design"
        ]
    }

    # 모든 단어와 카테고리 데이터프레임 생성
    words = [word for category in categories.values() for word in category]
    category_labels = [cat for cat, words_list in categories.items() for _ in words_list]
    df = pd.DataFrame({'Word': words, 'Category': category_labels})

    # 임베딩 및 차원 축소
    embeddings = get_embeddings(df['Word'].tolist())
    embedding = apply_umap(embeddings)

    # 결과를 데이터프레임에 저장
    df['UMAP_1'] = embedding[:, 0]
    df['UMAP_2'] = embedding[:, 1]

    # Streamlit UI
    st.subheader("🪄 Word suggestions by algorithm")
    selected_categories = st.multiselect(
        "Choose the algorithm you're interested in!",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )

    # 선택한 카테고리 필터링
    filtered_df = df[df['Category'].isin(selected_categories)]

    # 기본 색상 팔레트를 이용한 카테고리별 색상 매핑
    # category_colors = {cat: color for cat, color in zip(categories, DEFAULT_PLOTLY_COLORS)}
    # 카테고리별로 색상을 매핑 (Set1 팔레트 사용)
    category_colors = {cat: color for cat, color in zip(categories, qualitative.Dark2)}

    # Plotly 시각화를 위한 데이터 생성
    fig = Figure()
    # category_colors = {
    #     "DFS": "aquamarine", "BFS": "tomato", "Sort": "lightgreen", 
    #     "Greedy": "plum", "DP": "lightsalmon", "Shortest Path": "chocolate"
    # }

    for category in filtered_df['Category'].unique():
        category_df = filtered_df[filtered_df['Category'] == category]
        fig.add_trace(Scatter(
            x=category_df['UMAP_1'],
            y=category_df['UMAP_2'],
            mode='markers',
            # marker=dict(size=8, color=category_colors[category]),
            marker=dict(color=category_colors[category], size=10),
            # opacity=0.5,
            name=category,
            text=category_df['Word'],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    # 그래프 레이아웃 설정
    fig.update_layout(
        title=None,  # 제목 제거
        xaxis_title=None,
        yaxis_title=None,
        height=450,
        width=480,
        clickmode='event+select',
        legend=dict(
            orientation="h",  # 수평으로 정렬
            yanchor="bottom",
            y=-0.3,  # 그래프 아래로 이동
            xanchor="center",
            x=0.5
        )
    )
    # 그래프 축 숨기기
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # Streamlit CSS 스타일로 왼쪽 정렬
    st.markdown(
        """
        <style>
        .plot-container {
            display: flex;
            justify-content: flex-start; /* 왼쪽 정렬 */
            margin-left: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # 그래프 표시
    # st.plotly_chart(fig, use_container_width=True)
    clicked_points = plotly_events(fig, click_event=True, hover_event=False)

    