import streamlit as st
# from back.llm_service import get_huggingface_response
# from back.chat_storage import save_chat_history
# from back.explainability import compute_lime_values
# from front.visualization import display_lime_visualization

# 알고리즘별 시스템 프롬프트
ALGORITHM_PROMPTS = {
    "Depth-First Search(DFS)": """
            You are an algorithm implementation expert.
            The user wants to implement the Depth-First Search (DFS) algorithm.
            Follow these rules:

            Write the DFS algorithm in Python.
            The DFS algorithm should operate based on a graph represented using an adjacency list.
            Start the traversal from the given start node and return the visited nodes in order.
            Name the function dfs_traversal and ensure it takes the following parameters:
            graph: A graph in adjacency list format (dictionary).
            start: The node where the traversal should begin.
            The function should return a list of nodes in the order they were visited.
            The input graph can either be a connected graph or a disconnected graph.
            Add simple comments to explain the key steps in the code.
            Additionally, consider the following:

            Ensure visited nodes are not processed more than once.
            If the input graph is empty, return an empty list.
            Choose either a recursive or stack-based implementation.
            """,
    "Breadth-First Search(BFS)": """
            You are an algorithm implementation expert.
            The user wants to implement the Breadth-First Search (BFS) algorithm.
            Follow these rules:

            Write the BFS algorithm in Python.
            The BFS algorithm should operate based on a graph represented using an adjacency list.
            Start the traversal from the given start node and return the visited nodes in order.
            Name the function bfs_traversal and ensure it takes the following parameters:
            graph: A graph in adjacency list format (dictionary).
            start: The node where the traversal should begin.
            The function should return a list of nodes in the order they were visited.
            The input graph can either be a connected graph or a disconnected graph.
            Add simple comments to explain the key steps in the code.
            Additionally, consider the following:

            Ensure visited nodes are not processed more than once.
            If the input graph is empty, return an empty list.
            Use a queue to implement the traversal process.
            """ ,
    "Sort Algorithm": """
            You are an algorithm implementation expert.
            The user wants to implement a sorting algorithm.
            Follow these rules:

            Write the sorting algorithm in Python.
            The user wants to implement a specific sorting algorithm (e.g., Bubble Sort, Quick Sort, Merge Sort, etc.).
            The sorting algorithm implementation must follow these rules:
            The function name should be sort_algorithm and should take the list to be sorted as a parameter.
            The function must return the sorted list.
            The function should perform sorting in ascending order by default.
            Add simple comments to explain the key steps of the algorithm.
            Additionally, consider the following:

            If the input list is empty or contains only one element, return it as is.
            Choose and implement an appropriate sorting algorithm (e.g., Bubble Sort using basic loops).
            """,
    "Greedy Algorithm": """
            You are an expert in algorithm implementation.  
            The user wants to implement a Greedy Algorithm.  
            Please follow these rules:

            1. The Greedy Algorithm must be written in Python.
            2. The user is trying to solve a specific problem type (e.g., Activity Selection Problem, Minimum Spanning Tree, Coin Change Problem).
            3. The written function must include:
            - A function name and parameters that are flexibly set depending on the problem.
            - A clear explanation of the selection criteria used by the Greedy Algorithm (e.g., maximum, minimum, etc.).
            4. Include comments explaining the key steps.

            Additionally, please consider the following:
            - Clearly explain when the Greedy Algorithm guarantees the optimal solution.
            - Include exception handling for input values related to the problem.
            """,
    "Dynamic Programming(DP)": """
            You are an expert in algorithm implementation.  
            The user wants to solve a problem using Dynamic Programming (DP).  
            Please follow these rules:

            1. The Dynamic Programming algorithm must be written in Python.
            2. The user is trying to solve a specific problem type (e.g., Fibonacci sequence, Knapsack Problem, Shortest Path Problem).
            3. The written function must include:
            - A function name and parameters that are flexibly set depending on the problem.
            - Use one of the two approaches: `Memoization` or `Tabulation`.
            4. Include comments explaining the key steps and the structure of the DP table.

            Additionally, please consider the following:
            - Clearly implement the recurrence relation process.
            - Optimize the algorithm's time complexity.
            - Include exception handling for cases where the input data is empty.
            """,
    "Shortest Path Algorithm": """
            You are an expert in algorithm implementation.  
            The user wants to implement a shortest path algorithm.  
            Please follow these rules:

            1. The shortest path algorithm must be written in Python.
            2. The algorithm to be implemented is one of the following: Dijkstra, Floyd-Warshall, or Bellman-Ford.
            3. The written function must follow these rules:
            - The function name should be `shortest_path` and should take the input graph and starting node as parameters.
            - The graph should be represented as an adjacency list or a weighted matrix.
            - The function should return the shortest distance values for each node.
            4. Include comments explaining the key steps.

            Additionally, please consider the following:
            - Perform appropriate exception handling when the graph is empty.
            - Optimize the implemented algorithm for time complexity.
            """
}

def render_main_page():
    """중앙 메인 페이지 구현"""
    st.header(f"Conversation {st.session_state['current_page']}")

    # 안내문구 표시 (처음 한 번만)
    if not st.session_state.greetings:
        with st.chat_message("assistant"):
            intro = """
            Welcome to **Prompt Explainer**! 🤵🏻‍♀️\n
            This tool is designed to help you leverage LLMs (Large Language Models) more effectively when **solving algorithm problems**. ⛳️\n
            By visually highlighting which **parts of the prompt the LLM focuses on**, you can craft **better prompts** and receive **higher-quality response codes**. 🎲\n
            When you input a prompt, we will visualize the emphasized sections based on **SHAP values**. This allows you to learn better **prompt-writing strategies** and **maximize the utility of LLMs** in your workflow. 🎞️\n 
            Give it a try and enhance your experience in solving algorithmic problems! 🎸
            """
            st.markdown(intro)
            st.session_state.messages.append({"role": "assistant", "content": intro})  # 대화 기록에 추가
        st.session_state.greetings = True  # 상태 업데이트
        st.rerun()
        
    # 상태 관리: 버튼이 눌리지 않았을 때
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = None
        st.session_state.system_prompt = None
    # 대화 메시지 출력
    # st.subheader("Conversation")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 버튼이 눌리지 않았을 때 알고리즘 버튼 출력
    if st.session_state.button_pressed is None:
        # st.subheader("Choose an Algorithm")
        cols = st.columns(3)  # 3열 레이아웃
        for idx, (algo, prompt) in enumerate(ALGORITHM_PROMPTS.items()):
            with cols[idx % 3]:
                if st.button(algo):
                    st.session_state.button_pressed = algo  # 선택된 버튼을 상태로 저장
                    st.session_state.system_prompt = prompt  # 해당 시스템 프롬프트 저장
                    st.rerun()  # 페이지를 리프레시하여 새로운 버튼 상태 반영

    # 버튼이 눌린 후 선택된 알고리즘의 프롬프트 표시
    else:
        st.sidebar.title("🛠️ System Prompt")
        st.sidebar.info(st.session_state.system_prompt)  # 사이드바에 시스템 프롬프트 표시

        # 선택된 알고리즘의 상태에서 "다시 선택" 버튼을 만들어 상태 초기화
        col = st.columns(1)[0]  # 중앙 정렬을 위해 1열 사용
        with col:
            if st.button(st.session_state.button_pressed, key="selected_button"):
                st.session_state.button_pressed = None  # 상태 초기화
                st.session_state.system_prompt = None

        # "다시 선택" 버튼을 눌러 선택을 초기화하는 기능
        if st.button("back"):
            st.session_state.button_pressed = None  # 상태 초기화
            st.session_state.system_prompt = None
            st.rerun()  # 리프레시하여 다시 처음 상태로 돌아가기
    
    # 프롬프트 입력 창 (st.chat_input() 사용)
    user_input = st.chat_input("Enter your prompt!")
    if user_input:  # 사용자가 입력을 하면
        if user_input.strip():
            # LLM 응답 생성
            # response = get_huggingface_response(st.session_state["model"], user_input)
            # JSON 파일에서 SHAP 값 로드
            def load_shap_values(json_file):
                with open(json_file, 'r') as file:
                    data = json.load(file)
                return data['tokens'], data['shap_values']

            # JSON 파일 경로
            json_file_path = "shap_values.json"  # 파일 경로를 입력하세요
            tokens, shap_values = load_shap_values(json_file_path)

            # prompt 기여도 계산
            token_html = ""
            for token, shap_value in zip(tokens, shap_values):
                intensity = min(max(shap_value, 0), 1)  # SHAP 값을 0~1 범위로 정규화
                color = f"rgba(255, 0, 0, {intensity})"  # 빨간색 계열로 SHAP 값 표시
                token_html += f'<span style="background-color: {color}; padding: 2px; margin: 1px; border-radius: 4px;">{token}</span> '

            # 사용자 입력 시각화
            st.markdown(f"**User Input:**")
            st.markdown(token_html, unsafe_allow_html=True)

            # 모델 응답 출력
            # st.markdown(f"**LLM Response:**")
            # st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 8px;'>{response}</div>", unsafe_allow_html=True)

            # 메시지 기록 추가
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

            # 기여도 시각화

            # 대화 기록 저장
            chat_history = st.session_state["chat_history"]
            current_page = st.session_state["current_page"] - 1

            # 기존 페이지 업데이트
            if current_page < len(chat_history):
                chat_history[current_page] = {"messages": st.session_state["messages"]}
            else:
                chat_history.append({"messages": st.session_state["messages"]})

            # save_chat_history(chat_history)

            # UI 업데이트
            st.rerun()
    