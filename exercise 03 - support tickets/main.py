import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


df = pd.read_csv("tickets de suporte.csv", encoding='utf-8')

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("-- Informações do dataset: --")
print(f"• Shape: {df.shape}")
print(f"• Colunas: {list(df.columns)}")
print(f"• Tipos de dados:\n{df.dtypes.value_counts()}")


# ========================
# Etapa 1 – Exploração dos Dados
# ========================

def plot_countplot(data, column, title, xlabel, ylabel="Quantidade", rotation=0):
    plt.figure(figsize=(10, 6))
    order = data[column].value_counts().index
    ax = sns.countplot(x=column, data=data, order=order, palette="viridis")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


print("\n" + "=" * 50)
print(" ANÁLISE EXPLORATÓRIA")
print("=" * 50)

if "day_of_week" in df.columns:
    print("\n📌 Tickets por dia da semana:")
    day_counts = df["day_of_week"].value_counts()
    print(day_counts)
    plot_countplot(df, "day_of_week", "Distribuição de Tickets por Dia da Semana", "Dia da semana")

if "region" in df.columns:
    print("\n📌 Tickets por região:")
    region_counts = df["region"].value_counts()
    print(region_counts)
    plot_countplot(df, "region", "Distribuição de Tickets por Região", "Região")

# ========================
# Etapa 2 – Preparação Simples
# ========================

print("\n" + "=" * 50)
print(" PREPARAÇÃO DE DADOS")
print("=" * 50)

print("\n📌 Valores ausentes por coluna:")
missing_data = df.isna().sum()
missing_percent = (df.isna().sum() / len(df)) * 100
missing_info = pd.DataFrame({
    'Valores Ausentes': missing_data,
    'Percentual (%)': missing_percent.round(2)
})
print(missing_info[missing_info['Valores Ausentes'] > 0])

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    if df[col].skew() > 1:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mean())

for col in categorical_cols:
    if df[col].nunique() < 20:  # poucas categorias
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    else:
        df[col] = df[col].fillna("Unknown")

print("\n--- Dados após tratamento de missing values:")
print(f"Valores ausentes restantes: {df.isna().sum().sum()}")

# ========================
# Etapa 3 – Descoberta de Padrões
# ========================

print("\n" + "=" * 50)
print("🔍 ANÁLISE DE PADRÕES")
print("=" * 50)


def plot_grouped_analysis(data, group_col, value_col, title, palette="coolwarm"):
    grouped_data = data.groupby(group_col)[value_col].agg(['mean', 'count', 'std']).round(2)
    grouped_data = grouped_data.sort_values('mean', ascending=False)

    print(f"\n Estatísticas de {value_col} por {group_col}:")
    print(grouped_data)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=grouped_data.index, y=grouped_data['mean'], palette=palette)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(f"Média de {value_col}")
    plt.xticks(rotation=45)

    for i, (mean, count) in enumerate(zip(grouped_data['mean'], grouped_data['count'])):
        ax.text(i, mean + 0.1, f'{mean:.2f}\n(n={count})',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

if "customer_tier_cat" in df.columns and "past_90d_incidents" in df.columns:
    plot_grouped_analysis(df, "customer_tier_cat", "past_90d_incidents",
                          "Média de Incidentes (90d) por Customer Tier")

if "industry_cat" in df.columns and "past_90d_incidents" in df.columns:
    plot_grouped_analysis(df, "industry_cat", "past_90d_incidents",
                          "Média de Incidentes (90d) por Indústria", "mako")

if "past_90d_incidents" in df.columns:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df["past_90d_incidents"], bins=30, kde=True, color="orange")
    plt.title("Distribuição de Incidentes (90 dias)")
    plt.xlabel("Número de incidentes")

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df["past_90d_incidents"])
    plt.title("Boxplot de Incidentes (90 dias)")

    plt.tight_layout()
    plt.show()

    print("\n📊 Estatísticas descritivas dos incidentes (90 dias):")
    print(df["past_90d_incidents"].describe())

# ========================
# Extra – Análise Comparativa e Clusterização
# ========================

print("\n" + "=" * 60)
print("🔍 ANÁLISE COMPARATIVA E CLUSTERIZAÇÃO")
print("=" * 60)

# ========================
# Análise de Prioridade vs Variáveis
# ========================

print("\n ANÁLISE DE PRIORIDADE vs VARIÁVEIS CHAVE")

priority_column = None
for col in ['priority', 'severity', 'criticality']:
    if col in df.columns:
        priority_column = col
        break

if priority_column:
    print(f"✅ Coluna de prioridade encontrada: '{priority_column}'")
    print(f" Valores únicos: {df[priority_column].unique()}")

    if 'customer_tier_cat' in df.columns:
        print(f"\n📈 {priority_column.upper()} vs CUSTOMER TIER:")

        cross_tier = pd.crosstab(df[priority_column], df['customer_tier_cat'],
                                 normalize='columns') * 100

        print(" Distribuição percentual por tier:")
        print(cross_tier.round(1))

        # ----- gráfico
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='customer_tier_cat', hue=priority_column,
                      palette='RdYlGn_r', order=df['customer_tier_cat'].value_counts().index)
        plt.title(f'Distribuição de {priority_column} por Customer Tier', fontweight='bold')
        plt.xlabel('Customer Tier')
        plt.ylabel('Quantidade de Tickets')
        plt.legend(title=priority_column.title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print(f"\n Estatísticas de {priority_column} por tier:")
        tier_priority_stats = df.groupby('customer_tier_cat')[priority_column].value_counts(
            normalize=True).unstack().round(3)
        print(tier_priority_stats * 100)

    if 'region' in df.columns:
        print(f"\n REGIÃO vs MÉTRICAS DE PERFORMANCE:")

        time_metrics = []
        for metric in ['downtime', 'resolution_time', 'response_time', 'time_to_resolve']:
            if metric in df.columns:
                time_metrics.append(metric)

        if time_metrics:
            print(f"📊 Métricas de tempo disponíveis: {time_metrics}")

            for metric in time_metrics[:2]:
                plt.figure(figsize=(12, 6))

                # Boxplot por região
                sns.boxplot(data=df, x='region', y=metric, palette='Set2')
                plt.title(f'{metric.title()} por Região', fontweight='bold')
                plt.xlabel('Região')
                plt.ylabel(metric.title())
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

                region_stats = df.groupby('region')[metric].agg(['mean', 'median', 'std', 'count']).round(2)
                print(f"\n Estatísticas de {metric} por região:")
                print(region_stats)

                if metric == 'downtime' and 'APAC' in df['region'].unique():
                    apac_mean = df[df['region'] == 'APAC'][metric].mean()
                    other_mean = df[df['region'] != 'APAC'][metric].mean()
                    print(f"\n🔍 Comparação APAC vs outras regiões:")
                    print(f"   • APAC mean {metric}: {apac_mean:.2f}")
                    print(f"   • Outras regiões mean {metric}: {other_mean:.2f}")
                    print(f"   • Diferença: {abs(apac_mean - other_mean):.2f}")



# ========================
# Clusterização com Múltiplas Variáveis
# ========================

print("\n" + "=" * 50)
print(" CLUSTERIZAÇÃO COM MÚLTIPLAS VARIÁVEIS")
print("=" * 50)

cluster_candidates = [
    'company_size', 'past_90d_incidents', 'org_users',
    'past_30d_incidents', 'day_of_week_num'
]

available_cluster_vars = [var for var in cluster_candidates if var in df.columns]
print(f"Variáveis disponíveis para clusterização: {available_cluster_vars}")

if len(available_cluster_vars) >= 2:
    selected_vars = available_cluster_vars[:3] if len(available_cluster_vars) >= 3 else available_cluster_vars
    print(f"--- Variáveis selecionadas: {selected_vars}")

    X = df[selected_vars].copy()

    for col in selected_vars:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X_clean = X.dropna()

    if len(X_clean) >= 10:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        wcss = []
        max_clusters = min(8, len(X_clean) - 1)

        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
        plt.title('Número Ótimo de Clusters')
        plt.xlabel('Número de Clusters')
        plt.ylabel('WCSS')
        plt.xticks(range(1, max_clusters + 1))
        plt.grid(True, alpha=0.3)
        plt.show()

        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        df.loc[X_clean.index, 'cluster'] = clusters

        print(f"\n CARACTERÍSTICAS DOS {n_clusters} CLUSTERS:")

        cluster_analysis = df.groupby('cluster')[selected_vars].agg(['mean', 'count']).round(2)
        print(cluster_analysis)

        if len(selected_vars) == 2:
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_clean.iloc[:, 0], X_clean.iloc[:, 1],
                                  c=clusters, cmap='viridis', alpha=0.7, s=50)
            plt.xlabel(selected_vars[0])
            plt.ylabel(selected_vars[1])
            plt.title(f'Clusters: {selected_vars[0]} vs {selected_vars[1]}')
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True, alpha=0.3)
            plt.show()

        elif len(selected_vars) >= 3:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(X_clean.iloc[:, 0], X_clean.iloc[:, 1], X_clean.iloc[:, 2],
                                 c=clusters, cmap='viridis', alpha=0.7, s=50)

            ax.set_xlabel(selected_vars[0])
            ax.set_ylabel(selected_vars[1])
            ax.set_zlabel(selected_vars[2])
            ax.set_title(f'Clusters 3D: {selected_vars[0]}, {selected_vars[1]}, {selected_vars[2]}')
            plt.colorbar(scatter, label='Cluster')
            plt.show()

        print("\n PERFIL DOS CLUSTERS IDENTIFICADOS:")
        for cluster_id in range(n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            print(f"\n🔵 CLUSTER {cluster_id} ({len(cluster_data)} tickets):")

            for var in selected_vars:
                mean_val = cluster_data[var].mean()
                print(f"   • {var}: {mean_val:.2f} (média)")

            # por categorias
            if 'customer_tier_cat' in df.columns:
                top_tier = cluster_data['customer_tier_cat'].value_counts().head(1)
                if not top_tier.empty:
                    print(f"   • Customer Tier predominante: {top_tier.index[0]}")

            if 'industry_cat' in df.columns:
                top_industry = cluster_data['industry_cat'].value_counts().head(1)
                if not top_industry.empty:
                    print(f"   • Indústria predominante: {top_industry.index[0]}")

    else:
        print(" Dados insuficientes para clusterização após limpeza")
else:
    print("  Variáveis insuficientes para clusterização")

print("\n" + "=" * 60)
print("✅ ANÁLISE COMPARATIVA E CLUSTERIZAÇÃO CONCLUÍDA")
print("=" * 60)