import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="风电50年一遇极大风速计算器", layout="wide")
st.title("🌪️ 风电 50年一遇极大风速综合计算器")
st.markdown("包含：**五日雷暴法 (Gumbel)**、**5倍平均风速法**、**1/0.8倍年最大风速法**，并自动折算至标准空气密度。")

# ================= 侧边栏配置（文件解析参数） =================
st.sidebar.header("⚙️ 数据与参数设置")
sep_choice = st.sidebar.selectbox(
    "TXT 文件分隔符",
    ["自动识别 (推荐)", "制表符 (Tab)", "空格", "逗号 (,)"]
)
sep_map = {"自动识别 (推荐)": None, "制表符 (Tab)": "\t", "空格": r"\s+", "逗号 (,)": ","}
selected_sep = sep_map[sep_choice]

skip_rows = st.sidebar.number_input(
    "跳过表头行数 (Windographer 默认 13 行)",
    min_value=0, max_value=50, value=13, step=1
)

# ================= 主界面：手填当地空气密度 =================
st.subheader("🌡️ 当地空气密度（手填）")
col_a, col_b = st.columns(2)
with col_a:
    rho_local = st.number_input(
        "当地空气密度 (kg/m³)",
        min_value=0.8, max_value=1.4, value=1.225,
        step=0.001, format="%.3f",
        help="用于将计算结果折算至标准空气密度 (1.225 kg/m³)"
    )
with col_b:
    st.markdown(
        f"""
        <div style='padding-top: 28px;'>
        当前手填密度：<b>{rho_local:.3f} kg/m³</b><br>
        标准空气密度：<b>1.225 kg/m³</b><br>
        折算系数 √(ρ/1.225)：<b>{np.sqrt(rho_local / 1.225):.4f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ================= 文件上传 =================
uploaded_file = st.file_uploader("请上传测风数据 TXT 文件", type=['txt'])

if uploaded_file is not None:
    try:
        # ================= 核心防错读取逻辑 =================
        try:
            text_content = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            text_content = uploaded_file.getvalue().decode("gbk", errors="ignore")

        lines = text_content.splitlines()

        if len(lines) <= skip_rows:
            st.error("文件行数少于跳过的表头行数，请检查文件。")
            st.stop()

        data_lines = lines[skip_rows:]

        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=selected_sep,
            engine='python',
            header=None,
            on_bad_lines='skip'
        )

        st.write("📄 **解析后的前 18 行预览** (此时第一行应该是 2019-08-01 00:00):")
        st.dataframe(df.head(18))
        st.success(f"✅ 成功读取并清洗数据！有效计算行数：{len(df)} 行。")

        # 4. 列映射
        st.write("---")
        st.subheader("🔍 列映射设置")
        col_names = [f"第 {i+1} 列" for i in range(df.shape[1])]

        col1, col2 = st.columns(2)
        with col1:
            time_col = st.selectbox("选择【时间】所在的列", col_names, index=0)
        with col2:
            default_wind_idx = 1 if len(col_names) > 1 else 0
            wind_col = st.selectbox("选择【风速】所在的列", col_names, index=default_wind_idx)

        time_idx = col_names.index(time_col)
        wind_idx = col_names.index(wind_col)

        # 5. 数据处理
        df_clean = df[[time_idx, wind_idx]].copy()
        df_clean.columns = ['Time', 'WindSpeed']

        df_clean['Time'] = pd.to_datetime(df_clean['Time'], errors='coerce')
        df_clean['WindSpeed'] = pd.to_numeric(df_clean['WindSpeed'], errors='coerce')

        df_clean = df_clean.dropna(subset=['Time', 'WindSpeed'])
        df_clean = df_clean[(df_clean['WindSpeed'] > 0) & (df_clean['WindSpeed'] < 100)]

        if len(df_clean) == 0:
            st.error("❌ 数据清洗后为空！请检查时间格式或列是否选对。")
            st.stop()

        # ================= 方法一：5倍平均风速法 =================
        avg_wind = df_clean['WindSpeed'].mean()
        v_5avg_local = avg_wind * 5

        # ================= 方法二：1/0.8倍年最大风速法 =================
        max_wind = df_clean['WindSpeed'].max()
        v_08max_local = max_wind / 0.8

        # ================= 方法三：五日雷暴法 (Gumbel) =================
        start_time = df_clean['Time'].min()
        end_time = df_clean['Time'].max()
        df_clean['Period'] = ((df_clean['Time'] - start_time).dt.total_seconds() / (24 * 3600) // 5).astype(int) + 1

        extremes = df_clean.groupby('Period')['WindSpeed'].max().values
        N_total = len(extremes)          # 实时总样本数：用于 C1、C2

        # 每年五日极值个数 n_y：默认 73；更严谨可用 N_total / 数据年数
        data_years = (end_time - start_time).total_seconds() / (365.25 * 24 * 3600)
        N_YEAR = N_total / data_years if data_years > 0 else 73.0
        # 若想严格固定为一年 73 个，可改成：N_YEAR = 73.0

        param_dict = {
            10: (0.94970, 0.49520), 15: (1.02057, 0.51820), 20: (1.06283, 0.52355),
            25: (1.09145, 0.53086), 30: (1.11238, 0.53622), 35: (1.12847, 0.54034),
            40: (1.14132, 0.54362), 45: (1.15185, 0.54630), 50: (1.16066, 0.54853),
            60: (1.17465, 0.55208), 70: (1.18536, 0.55477), 80: (1.19385, 0.55688),
            90: (1.20649, 0.55860), 100: (1.20649, 0.56002), 250: (1.24292, 0.56878),
            500: (1.25880, 0.57240), 1000: (1.26851, 0.57450)
        }

        n_keys = np.array(list(param_dict.keys()))
        c1_vals = np.array([v[0] for v in param_dict.values()])
        c2_vals = np.array([v[1] for v in param_dict.values()])

        # C1、C2 用实时总样本数 N_total
        if N_total > 1000:
            C1, C2 = 1.28255, 0.57722
        elif N_total < 10:
            C1, C2 = 0.94970, 0.49520
        else:
            C1 = np.interp(N_total, n_keys, c1_vals)
            C2 = np.interp(N_total, n_keys, c2_vals)

        mu = np.mean(extremes)
        sigma = np.std(extremes, ddof=1)

        if sigma > 0:
            alpha = C1 / sigma
            u = mu - (C2 / alpha)
            # 关键修改：ln 里用每年样本数 N_YEAR，而不是 N_total
            v_5day_local = u - (1 / alpha) * np.log(np.log((50 * N_YEAR) / (50 * N_YEAR - 1)))
        else:
            v_5day_local = 0
            st.warning("⚠️ 极值样本方差为0，五日雷暴法计算失败。")

        # ================= 标准空气密度换算 =================
        ratio = np.sqrt(rho_local / 1.225)

        v_5avg_std = v_5avg_local * ratio
        v_08max_std = v_08max_local * ratio
        v_5day_std = v_5day_local * ratio

        v_max_all_local = max(v_5avg_local, v_08max_local, v_5day_local)
        v_max_all_std = max(v_5avg_std, v_08max_std, v_5day_std)

        # ================= 展示结果 =================
        st.write("---")
        st.header("🎯 综合计算结果")

        st.markdown(f"**当前当地空气密度:** `{rho_local:.3f} kg/m³` | **标准空气密度:** `1.225 kg/m³` | **折算系数:** `{ratio:.4f}`")

        result_data = {
            "计算方法": ["1. 5倍平均风速法", "2. 1/0.8倍最大风速法", "3. 五日雷暴法 (Gumbel)"],
            "当地密度下风速 (m/s)": [f"{v_5avg_local:.2f}", f"{v_08max_local:.2f}", f"{v_5day_local:.2f}"],
            "标准密度下风速 (m/s)": [f"{v_5avg_std:.2f}", f"{v_08max_std:.2f}", f"{v_5day_std:.2f}"]
        }
        st.table(pd.DataFrame(result_data))

        st.success(f"🏆 **最终推荐设计值 (三者取大)**：当地密度下 **{v_max_all_local:.2f} m/s** | 标况下 **{v_max_all_std:.2f} m/s**")

        with st.expander("📊 查看五日雷暴法详细参数 (N_total, N_YEAR, C1, C2, μ, σ, α, u)"):
            st.write(f"- **数据时间范围**: {start_time} 至 {end_time}")
            st.write(f"- **数据年数**: {data_years:.3f} 年")
            st.write(f"- **五日极值总样本数 (N_total)**: {N_total}（用于 C1、C2）")
            st.write(f"- **每年五日极值数 (N_YEAR)**: {N_YEAR:.3f}（用于 ln 函数）")
            st.write(f"- **无偏修正系数 (C1)**: {C1:.6f}")
            st.write(f"- **无偏修正系数 (C2)**: {C2:.6f}")
            if sigma > 0:
                st.write(f"- **样本均值 (μ)**: {mu:.3f} m/s")
                st.write(f"- **样本标准差 (σ)**: {sigma:.3f} m/s")
                st.write(f"- **尺度参数 (α)**: {alpha:.4f}")
                st.write(f"- **位置参数 (u)**: {u:.3f} m/s")
            st.write(f"- **年平均风速**: {avg_wind:.3f} m/s")
            st.write(f"- **实测最大风速**: {max_wind:.3f} m/s")

    except Exception as e:
        st.error(f"⚠️ 处理文件时出错：{e}")
