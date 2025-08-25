import java.io.*;
import java.util.*;

public class IrisKNN {

    static class Amostra {
        double[] x; 
        int y;
        Amostra(double[] x, int y) { this.x = x; this.y = y; }
    }

    static char detectarDelimitador(String header) {
        int c = 0, p = 0;
        for (char ch : header.toCharArray()) {
            if (ch == ',') c++;
            else if (ch == ';') p++;
        }
        return (p > c) ? ';' : ',';
    }

    static List<String> parseCSVLine(String line, char delim) {
        List<String> out = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);
            if (ch == '"') {
                inQuotes = !inQuotes;
            } else if (ch == delim && !inQuotes) {
                out.add(sb.toString());
                sb.setLength(0);
            } else {
                sb.append(ch);
            }
        }
        out.add(sb.toString());
        return out;
    }

    static double parseNumero(String s) {
        String t = s.trim().replace("\"", "");
        if (t.indexOf(',') >= 0 && t.indexOf('.') < 0) t = t.replace(',', '.');
        return Double.parseDouble(t);
    }

    static int mapClasse(String speciesRaw) {
        String s = speciesRaw.trim().toLowerCase();
        if (s.contains("setosa")) return 0;
        if (s.contains("versicolor")) return 1;
        // padrão
        return 2; // virginica
    }

    static String[] classNames = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    static List<Amostra> carregarIrisCSV(String path) throws Exception {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String header = br.readLine();
            if (header == null) throw new RuntimeException("CSV vazio.");
            char delim = detectarDelimitador(header);
            List<String> h = parseCSVLine(header, delim);

            // mapear colunas por nome
            Map<String, Integer> idx = new HashMap<>();
            for (int i = 0; i < h.size(); i++) {
                idx.put(h.get(i).trim().toLowerCase(), i);
            }

            // tentar achar os nomes esperados
            // permite "Id" opcional
            Integer iSepalLen = idx.get("sepallengthcm");
            Integer iSepalWid = idx.get("sepalwidthcm");
            Integer iPetalLen = idx.get("petallengthcm");
            Integer iPetalWid = idx.get("petalwidthcm");
            Integer iSpecies  = idx.get("species");

            if (iSepalLen == null || iSepalWid == null || iPetalLen == null || iPetalWid == null || iSpecies == null) {
                // fallback: alguns CSVs podem vir sem "Cm" nos nomes, apenas boas práticas
                iSepalLen = (iSepalLen != null) ? iSepalLen : idx.get("sepallength");
                iSepalWid = (iSepalWid != null) ? iSepalWid : idx.get("sepalwidth");
                iPetalLen = (iPetalLen != null) ? iPetalLen : idx.get("petallength");
                iPetalWid = (iPetalWid != null) ? iPetalWid : idx.get("petalwidth");
                // species já tentado
            }
            if (iSepalLen == null || iSepalWid == null || iPetalLen == null || iPetalWid == null || iSpecies == null) {
                throw new RuntimeException("Cabeçalho não encontrado. Esperado: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species");
            }

            List<Amostra> dados = new ArrayList<>();
            String line;
            int linhaNum = 1;
            while ((line = br.readLine()) != null) {
                linhaNum++;
                if (line.trim().isEmpty()) continue;
                List<String> parts = parseCSVLine(line, delim);

                try {
                    double sl = parseNumero(parts.get(iSepalLen));
                    double sw = parseNumero(parts.get(iSepalWid));
                    double pl = parseNumero(parts.get(iPetalLen));
                    double pw = parseNumero(parts.get(iPetalWid));
                    int y = mapClasse(parts.get(iSpecies));
                    dados.add(new Amostra(new double[]{sl, sw, pl, pw}, y));
                } catch (Exception e) {
                    // pula linhas problemáticas mas informa
                    System.err.println("Aviso: pulando linha " + linhaNum + " (erro ao parsear): " + e.getMessage());
                }
            }
            if (dados.isEmpty()) throw new RuntimeException("Nenhuma amostra válida lida.");
            return dados;
        }
    }

    static void padronizar(List<Amostra> treino, List<Amostra> teste) {
        int d = 4;
        double[] mean = new double[d];
        double[] var = new double[d];

        for (Amostra a : treino) for (int j = 0; j < d; j++) mean[j] += a.x[j];
        for (int j = 0; j < d; j++) mean[j] /= treino.size();
        for (Amostra a : treino) for (int j = 0; j < d; j++) var[j] += Math.pow(a.x[j] - mean[j], 2);
        for (int j = 0; j < d; j++) var[j] = Math.sqrt(var[j] / Math.max(1, treino.size() - 1));

        for (Amostra a : treino) for (int j = 0; j < d; j++) a.x[j] = (a.x[j] - mean[j]) / (var[j] == 0 ? 1 : var[j]);
        for (Amostra a : teste)  for (int j = 0; j < d; j++) a.x[j] = (a.x[j] - mean[j]) / (var[j] == 0 ? 1 : var[j]);
    }

    static double distancia(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    static int predizer(List<Amostra> treino, double[] x, int k) {
        int n = treino.size();
        Integer[] idx = new Integer[n];
        double[] dist = new double[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
            dist[i] = distancia(treino.get(i).x, x);
        }
        Arrays.sort(idx, Comparator.comparingDouble(i -> dist[i]));
        int[] votos = new int[3];
        for (int i = 0; i < k; i++) {
            votos[treino.get(idx[i]).y]++;
        }
        int best = 0;
        for (int c = 1; c < 3; c++) if (votos[c] > votos[best]) best = c;
        return best;
    }

    static void imprimirMatrizConfusao(int[][] cm) {
        System.out.println("\nMatriz de Confusão (linhas = real, colunas = previsto):");
        System.out.printf("%18s%18s%18s%n", classNames[0], classNames[1], classNames[2]);
        for (int i = 0; i < 3; i++) {
            System.out.printf("%-16s", classNames[i]);
            for (int j = 0; j < 3; j++) {
                System.out.printf("%18d", cm[i][j]);
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        try {

            String caminho = "Iris.csv";
            List<Amostra> dataset = carregarIrisCSV(caminho);
            System.out.println("Amostras lidas: " + dataset.size());

            Collections.shuffle(dataset, new Random(42));
            int n = dataset.size();
            int nTreino = (int) Math.round(n * 0.7);
            List<Amostra> treino = new ArrayList<>(dataset.subList(0, nTreino));
            List<Amostra> teste  = new ArrayList<>(dataset.subList(nTreino, n));

            padronizar(treino, teste);

            int k = Math.max(3, (int) Math.round(Math.sqrt(treino.size())));
            if (k % 2 == 0) k++; // evita empate
            System.out.println("Usando K = " + k);

            int corretos = 0;
            int[][] cm = new int[3][3];

            for (Amostra a : teste) {
                int yPred = predizer(treino, a.x, k);
                if (yPred == a.y) corretos++;
                cm[a.y][yPred]++;
            }

            double acuracia = 100.0 * corretos / teste.size();
            System.out.printf("✅ Acurácia: %.2f%%%n", acuracia);
            imprimirMatrizConfusao(cm);

        } catch (Exception e) {
            System.err.println("Erro ao executar: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
