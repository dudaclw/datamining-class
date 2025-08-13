import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv('alunos.csv')
dados['media'] = dados[['nota1', 'nota2', 'nota3']].mean(axis=1)
dados['situacao'] = dados['media'].apply(lambda x: 'Aprovado' if x >= 7 else 'Reprovado')

# Análises
media_turma = dados['media'].mean()
maior_nota = dados[['nota1', 'nota2', 'nota3']].max().max()
media_idade = dados['idade'].mean()
media_aprovados = dados[dados['situacao'] == 'Aprovado']['media'].mean()
media_reprovados = dados[dados['situacao'] == 'Reprovado']['media'].mean()
media_por_idade = dados.groupby('idade')['media'].mean()

# Contagem de aprovados e reprovados
contagem_situacao = dados['situacao'].value_counts()

# Exibe resultados
print(f"- Média da turma: {media_turma:.2f}")
print(f"- Maior nota: {maior_nota:.1f}")
print(f"- Média de idade: {media_idade:.2f} anos")
print(f"- Média dos aprovados: {media_aprovados:.2f}")
print(f"- Média dos reprovados: {media_reprovados:.2f}")
print("\n- Média de nota por idade:")
print(media_por_idade)
print("\n- Contagem de aprovados e reprovados:")
print(contagem_situacao)

# Gráfico 
fig, ax1 = plt.subplots(figsize=(9,5))

plt.figure(figsize=(8,5))
plt.bar(media_por_idade.index, media_por_idade.values, color="purple", edgecolor="black")
plt.title("Média de Nota por Idade", fontsize=12)
plt.xlabel("Idade", fontsize=12)
plt.ylabel("Média das Notas", fontsize=12)
plt.xticks(media_por_idade.index)
plt.ylim(0, 10)  # Escala fixa 0 a 10
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
