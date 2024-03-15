### 外部ライブラリをインポートする ###
from codecs import register_error
from lib2to3.pgen2.pgen import DFAState
import streamlit as st
import japanize_matplotlib
import matplotlib.pyplot as plt
import folium
import numpy as np
import pandas as pd
import math

### 自作のモジュールをインポートする ###
from modules import VRProblem
from modules import MOProblem

apikey = st.secrets["apikey"]

# CSSの設定をする関数
def set_font_style():

    st.markdown(
        """
        <style>
        textarea {
            font-size: 1.2rem !important;
            font-family:  monospace !important;
        }

        code {
            font-size: 1.2rem !important;
            font-family:  monospace !important;
        }

        div.stButton > button:first-child  {
            margin: 0 auto;
            max-width: 240px;
            padding: 10px 10px;
            background: #6bb6ff;
            color: #FFF;
            transition: 0.3s ease-in-out;
            font-weight: 600;
            border-radius: 100px;
            box-shadow: 0 5px 0px #4f96f6, 0 10px 15px #4f96f6;            
            border: none            
        }

        div.stButton > button:first-child:hover  {
            color: #FFF;
            background:#FF2F2F;
            box-shadow: 0 5px 0px #B73434,0 7px 30px #FF2F2F;
            border: none            
          }

        div.stButton > button:first-child:focus {
            color: #FFF;
            background: #6bb6ff;
            box-shadow: 0 5px 0px #4f96f6, 0 10px 15px #4f96f6;
            border: none            
          }


        button[title="View fullscreen"]{
            visibility: hidden;}

        </style>
        """,
        unsafe_allow_html=True,
    )

# 各種フラグなどを初期化する関数（最初の1回だけ呼ばれる）
def init_parameters():

    # タブに表示されるページ名の変更
    st.set_page_config(page_title="量子アルゴリズム", initial_sidebar_state="expanded", )

_colors = [
    "green",
    "orange",
    "blue",
    "red",
    "purple",
    "pink",
    "darkblue",
    "cadetblue",
    "darkred",
    "lightred",
    "darkgreen",
    "lightgreen",
    "lightblue",
    "darkpurple",
]

def plot_solution(coord: dict, title: str, best_tour: dict = dict()):
    l = len(coord)
    center = [
        sum(lat for _, lat in coord.values()) / l,
        sum(lon for lon, _ in coord.values()) / l,
    ]
    m = folium.Map(center, tiles="OpenStreetMap", zoom_start=10.5)
    folium.Marker(
        location=coord[0][::-1],
        popup=f"depot",
        icon=folium.Icon(icon="car", prefix="fa"),
    ).add_to(m)

    _color = _colors[1]
    if best_tour:
        for k, tour in best_tour.items():
            _color = _colors[k % len(_colors)]
            for city in tour:
                if city == 0:
                    continue

                folium.Marker(
                    location=coord[city][::-1],
                    popup=f"person{k}",
                    icon=folium.Icon(
                        icon="school", prefix="fa", color="white", icon_color=_color
                    ),
                ).add_to(m)
            folium.vector_layers.PolyLine(
                locations=[coord[city][::-1] for city in tour], color=_color, weight=3
            ).add_to(m)
    else:
        for k, node in coord.items():
            if k == 0:
                continue
            folium.Marker(
                location=node[::-1],
                popup=f"customer{k}",
                icon=folium.Icon(
                    icon="school", prefix="fa", color="white", icon_color=_color
                ),
            ).add_to(m)

    title = f"<h4>{title}</h4>"
    m.get_root().html.add_child(folium.Element(title))

    # 緯度経度の範囲を取得
    latitudes = [lat for _, lat in coord.values()]
    longitudes = [lon for lon, _ in coord.values()]
    m.fit_bounds([[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]])

    return m


#----------------------------------------------#
# シミュレーター画面の表示の処理群
#----------------------------------------------#
def view_mockup():

    # セレクトボックス（メインメニュー）
    menu = ['【選択してください】', '配送最適化', '献立最適化', '従業員割当問題']
    choice = st.sidebar.selectbox('モードを選択してください', menu)

    if menu.index(choice) == 0:
        st.info('左側のメニューから「モード」を選択してください')

    if menu.index(choice) == 1:
        st.title('容量制約つき運搬経路問題')
        # 画像のパス
        image_path = "assets/image/CapacitatedVehicleRoutingProblem.png"
        # 画像を表示
        st.image(image_path, caption="(Capacitated Vehicle Routing Problem, CVRP)", use_column_width=True)
        st.write('運搬経路(配送計画)問題とは、配送拠点(depot)から複数の需要地への配送を効率的に行おうとする配送ルート決定問題です。より具体的には、配送車両の総移動距離が最小になるような配送車両と需要地の割り当て、需要地の訪問順序を決定します。')
        st.write('このデモで取り扱う容量制約付き運搬経路問題は、上記運搬経路問題に各車両の積載上限が追加された問題です。つまり、各配送車両は積載量制約を満たした上で配送を行う必要があります。')
        st.write('今回は配送拠点(デポ)が一つかつ、各需要地の需要と車の容量が整数値のみを取るような場合を考えます。')
        st.write('運搬経路問題の具体的な応用先として、')
        st.write('・　郵便などの運送業における効率的な配送計画の策定')
        st.write('・　ごみ収集や道路清掃における訪問順序の決定')
        st.write('などがあります。')
        st.markdown("---")

        colA, colB = st.columns([3, 1])
        # スライダーを表示
        with colA:
            selected_value = st.slider('車の台数を選択してください', 1, 5, 3)
        with colB:
            # 実行ボタンを追加
            button_pressed = st.button('✔RUN')

        df_read = pd.read_csv("pos_data.csv")
        df_base = df_read.iloc[0:1]
        df_data = df_read.iloc[1:]
        unique_districts = df_data['District'].unique()

        col1, col2, col3, col4 = st.columns(4)
        selected_districts = {}
        for i, district in enumerate(unique_districts):
            count = df_data[df_data['District'] == district].shape[0]
            label = f"{district} ({count}件)"
            # 2列目までにチェックボックスを配置
            if i % 4 == 0:
                checkbox = col1.checkbox(label)
            elif i % 4 == 1:
                checkbox = col2.checkbox(label)
            elif i % 4 == 2:
                checkbox = col3.checkbox(label)
            else:
                checkbox = col4.checkbox(label)
            selected_districts[district] = checkbox

        selected_data = pd.concat([df_base, df_data[df_data['District'].isin([district for district, selected in selected_districts.items() if selected])]])
        ind2coord = selected_data.reset_index(drop=True).apply(lambda row: (row[3], row[2]), axis=1).to_dict()

        if button_pressed:
            VRProblem.set_api_key(apikey)
            best_tour = VRProblem.find_best_tour(selected_data[['Longitude', 'Latitude']].values.tolist(), selected_value)
        else:
            best_tour = None

        map_ = plot_solution(ind2coord, "title", best_tour)
        html_map = folium.Figure().add_child(map_).render()

        st.write(f'配送先：{len(ind2coord)-1}件')
        st.components.v1.html(html_map, height=500)

    if menu.index(choice) == 2:
        st.title('献立最適化問題')
        # 画像のパス
        image_path = "assets/image/MenuOptimization.png"
        # 画像を表示
        st.image(image_path, use_column_width=True)
        st.write('献立最適化問題とは、特定の制約や要求に基づいて、最も効率的かつ栄養価の高い食事を計画することを目的とした問題です。より具体的には、目標となる栄養素の必要量を満たす献立の組み合わせを決定します。')
        st.write('このデモで取り扱う献立最適化問題は、バランスの取れた食事を実現するための最適な組み合わせを見つけることに焦点を当てています。')
        st.write('献立最適化問題の具体的な応用先として、')
        st.write('・　家庭や学校の食事計画で、予算内で栄養バランスのとれたメニューの作成')
        st.write('・　特定の健康上の制約（例えば、糖尿病や高血圧など）を持つ人々のための個別の食事計画')
        st.write('・　顧客の好みや季節に応じた献立の提案')
        st.write('などがあります。')
        st.markdown("---")



        df_read = pd.read_csv("menu_data.csv")
        columns = df_read.columns
        unique_vals = df_read.iloc[:, 0].unique() # データ区分
        total_nutrients = [0] * (len(columns) - 2) # データ区分と料理名を除く
        goal_data = [0] * (len(columns) - 2)
        labels = list(columns[2:]) # データ区分と料理名を除く

        min_values = [math.floor(val * 3) for val in df_read.iloc[:, 2:].min()]
        max_values = [math.floor(val * 3) for val in df_read.iloc[:, 2:].max()]
        num_cols = len(total_nutrients)
        colsA = st.columns(num_cols)
        for col_index in range(num_cols):
            with colsA[col_index]:
                goal_data[col_index] = st.slider(labels[col_index], min_values[col_index], max_values[col_index], int(min_values[col_index] + ((max_values[col_index] + min_values[col_index]) / 3)))
                if col_index == num_cols - 1:
                    button_pressed = st.button('✔RUN')

        num_cols = len(unique_vals)
        cols = [st.columns(num_cols) for _ in range((len(unique_vals) + num_cols - 1) // num_cols)]
        selected_dish = {}
        for i, val in enumerate(unique_vals):
            col_index = i % num_cols
            with cols[i // num_cols][col_index]:
                selected_dish[val] = st.selectbox(f'{val}を選択してください:', df_read[df_read.iloc[:, 0]==val].iloc[:, 1])
                selected_row = df_read[(df_read.iloc[:, 0]==val) & (df_read.iloc[:, 1]==selected_dish[val])]

                # 選択されたデータの栄養素を表示
                st.write(selected_row.transpose()[2:])

                # 各栄養素の合計に追加
                for i in range(len(total_nutrients)):
                    total_nutrients[i] += selected_row.iloc[0, i + 2]

        if button_pressed:
            MOProblem.set_api_key(apikey)
            try:
                best_menu = MOProblem.find_best_menu(df_read, goal_data)
                if np.sum(np.where(best_menu == 1)[0]) == 0:
                    raise RuntimeError("制約条件を満たす組み合わせは見つかりませんでした")
            except:
                best_menu = None
                st.warning('制約条件を満たす組み合わせは見つかりませんでした')
        else:
            best_menu = None

        st.markdown("---")

        def normalize_data(data, max_values):
            return [d / max_val for d, max_val in zip(data, max_values)]

        # レーダーチャートの描画
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        data = normalize_data(total_nutrients, max_values) + [total_nutrients[0] / max_values[0]]
        g_data = normalize_data(goal_data, max_values) + [goal_data[0] / max_values[0]]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.fill(angles, data, color='blue', alpha=0.25)
        ax.plot(angles, g_data, color='red', linewidth=2)  # 目標値を追加
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_theta_zero_location('N')

        # タイトルを設定
        ax.set_title('栄養素比較')

        colA, colB = st.columns([1, 1])
        # スライダーを表示
        with colA:
            colAA, colAB = st.columns([1, 3])
            for v in unique_vals:
                with colAA:
                    st.write(f'{v} ：')
                with colAB:
                    st.write(selected_dish[v])
            st.pyplot(plt)
            st.write("■ 合計 ■")
            for i in range(len(labels)):
                st.write(f"{labels[i]}: {total_nutrients[i]} / {goal_data[i]}")

        if not best_menu is None:
            indices = np.where(best_menu == 1)[0]
            selected_rows = df_read.iloc[indices]
            total_nutrients2 = selected_rows.iloc[:, 2:].sum()
            total_nutrients2 = list(total_nutrients2.values)
            data2 = normalize_data(total_nutrients2, max_values) + [total_nutrients2[0] / max_values[0]]

            # レーダーチャートの描画
            plt.figure()
            ax = plt.subplot(111, polar=True)
            ax.fill(angles, data2, color='blue', alpha=0.25)
            ax.plot(angles, g_data, color='red', linewidth=2)  # 目標値を追加
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_theta_zero_location('N')

            # タイトルを設定
            ax.set_title('栄養素比較')

            with colB:
                colBA, colBB = st.columns([1, 3])
                for i in range(len(selected_rows)):
                    with colBA:
                        st.write(f'{selected_rows.iloc[i][0]} ：')
                    with colBB:
                        st.write(selected_rows.iloc[i][1])
                st.pyplot(plt)
                st.write("■ 合計 ■")
                for i in range(len(labels)):
                    st.write(f"{labels[i]}: {total_nutrients2[i]} / {goal_data[i]}")

    if menu.index(choice) == 3:
        st.title(menu[3])
        # 画像のパス
        image_path = "assets/image/EmployeeAssignmentProblem.png"
        # 画像を表示
        st.image(image_path, use_column_width=True)
        st.write('従業員割当問題とは、各従業員の役職やスキル、希望勤務地と、日々変動する多種多様な業務を考慮して、各店舗に適切に従業員を割り当てる問題です。')
        st.write('このデモで取り扱う従業員割当問題は、役職やスキルの種類やレベル、役割や希望勤務地に応じた適切な割り当てを行い、各店舗における業務の効率化と店舗間・従業員間の業務量の平準化を目的としています。')
        st.write('従業員割当問題の具体的な応用先として、')
        st.write('・　店舗スタッフの適切な配置と業務割り当て')
        st.write('・　製造業における工場ラインでの作業員配置と役割分担の決定')
        st.write('・　サービス業におけるレセプションやカスタマーサポートスタッフの日々のスケジュール管理')
        st.write('などがあります。')
        st.markdown("---")



#----------------------------------------------#
# メイン関数
#----------------------------------------------#
def main():

    ### 各種フラグなどを初期化する関数をコール ###
    init_parameters()

    # フォントスタイルの設定
    set_font_style()

    # メニューを表示する
    view_mockup()


if __name__ == "__main__":
    main()
