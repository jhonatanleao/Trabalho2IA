clear all;
dataset = dlmread('vinhoVermelho.csv', sep=",");
hiper_K = 4;
hiper_P = 2;
hiper_tam_treino = 2/3;
X = dataset(:, 1:end-1);
Y = dataset(:, end);
tam_treino1 = int32(hiper_tam_treino * length(X));
tam_teste = length(X) - tam_treino1;
tam_treino = int32(hiper_tam_treino * tam_treino1);
tam_validacao = tam_treino1 - tam_treino;
X = (X - mean(X)) ./ std(X);
r = randperm(length(X));
X_train = X(r(1:tam_treino), :);
Y_train = Y(r(1:tam_treino), :);
X_train_validation = X(r(tam_treino+1:tam_validacao+tam_treino), :);
Y_train_validation = Y(r(tam_treino+1:tam_validacao+tam_treino), :);
X_test = X(r(tam_validacao+tam_treino+1:end), :);
Y_test = Y(r(tam_validacao+tam_treino+1:end), :);
[autovetores, autovalores] = eig(cov(X_train));
autovalores = diag(autovalores);
[autovalores, indice] = sort(autovalores, 'descend');
autovetores = autovetores(:, indice(1:5));
X_train = X_train * autovetores;
X_train_validation = X_train_validation * autovetores;
X_test = X_test * autovetores;
primeiro = find( Y_train == 1);
segundo = find(Y_train == 2);
terceiro = find(Y_train == 3);
quarto = find(Y_train == 4);
figure();plot3(0,0,0);hold on;grid on;title('Dados de treino');
plot3(X_train(primeiro, 1), X_train(primeiro, 2), X_train(primeiro, 3), 'bo');
plot3(X_train(segundo, 1), X_train(segundo, 2), X_train(segundo, 3), 'go');
plot3(X_train(terceiro, 1), X_train(terceiro, 2), X_train(terceiro, 3), 'ro');
plot3(X_train(quarto, 1), X_train(quarto, 2), X_train(quarto, 3), 'mo');

dlmwrite('treino.mat', [X_train Y_train])
dlmwrite('validacao.mat', [X_train_validation Y_train_validation])
dlmwrite('teste.mat', [X_test Y_test])

resultados = zeros(tam_validacao,1);
k = hiper_K;
p = hiper_P;

for i=1:tam_validacao
    %Passo 2:
    distancia = abs(Y_train_validation(i,:) - X_train).^p;
    Lp = sum(distancia,2).^(1/p);
    [_, indices] = sort(Lp);

    %Passo 3:
    mais_prox = indices(1:k);

    %Passo 4:
    saidas = Y_train(mais_prox);
    votacao = mode(saidas);

    %Passo 5:
    resultados(i) = votacao;
end;

msg_resultados = sprintf('Resultados -- Acertos: %0.2f%%', sum(resultados == Y_train_validation)/double(tam_validacao)*100);
disp(msg_resultados);

teste_1 = find(Y_train_validation == 1);
teste_1 = [teste_1]
teste_2 = find(Y_train_validation == 2);
teste_2 = [teste_2]
teste_3 = find(Y_train_validation == 3);
teste_3 = [teste_3]
teste_4 = find(Y_train_validation == 4);
teste_4 = [teste_4]
resultado_1 = find(resultados == 1);
resultado_1 = [resultado_1]
resultado_2 = find(resultados ==  2);
resultado_2 = [resultado_2]
resultado_3 = find(resultados ==  3);
resultado_3 = [resultado_3]
resultado_4 = find(resultados ==  4);
resultado_4 = [resultado_4]
figure();
plot3(0,0,0);hold on;grid on;title(msg_resultados);
plot3( X_test(teste_1, 1), X_test(teste_1, 2), X_test(teste_1, 3), 'bo');
plot3( X_test(teste_2, 1), X_test(teste_2, 2), X_test(teste_2, 3), 'ro');
plot3( X_test(teste_3, 1), X_test(teste_3, 2), X_test(teste_3, 3), 'go');
plot3( X_test(teste_4, 1), X_test(teste_4, 2), X_test(teste_4, 3), 'mo');
plot3( X_test(resultado_1, 1), X_test(resultado_1, 2), X_test(resultado_1, 3), 'b.');
plot3( X_test(resultado_2, 1), X_test(resultado_2, 2), X_test(resultado_2, 3), 'r.');
plot3( X_test(resultado_3, 1), X_test(resultado_3, 2), X_test(resultado_3, 3), 'g.');
plot3( X_test(resultado_4, 1), X_test(resultado_4, 2), X_test(resultado_4, 3), 'm.');
