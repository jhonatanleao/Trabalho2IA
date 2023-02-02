for j=1:2000
    clear all;
    dataset = dlmread('vinhoVermelho.csv', sep=",");
    
    % hiperparâmetros
    hiper_K = randi([3, 5, 7, 9]);
    hiper_P = randi([1,2]);
    fracao_tam_treino = 2/3;
    num_dimensoes = randi([2,7])
    
    % separação dos datasets
    X = dataset(:, 1:end-1);
    Y = dataset(:, end);

    % divisão 
    subconjunto_treino = int32(fracao_tam_treino * length(X));
    tam_teste = length(X) - subconjunto_treino;
    tam_treino = int32(fracao_tam_treino * subconjunto_treino);
    tam_validacao = subconjunto_treino - tam_treino;

    % PCA
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
    
    autovetores = autovetores(:, indice(1:num_dimensoes));
    X_train = X_train * autovetores;
    X_train_validation = X_train_validation * autovetores;
    X_test = X_test * autovetores;
    primeiro = find( Y_train == 1);
    segundo = find(Y_train == 2);

    % figure();plot(0,0);hold on;grid on;title('Dados de treino');
    % plot(X_train(primeiro, 1), X_train(primeiro, 2), 'bo');
    % plot(X_train(segundo, 1), X_train(segundo, 2), 'ro');

    % KNN
    resultados = zeros(tam_validacao,1);
    k = hiper_K;
    p = hiper_P;

    for i=1:tam_validacao
        %Passo 2:
        distancia = abs(X_train_validation(i,:) - X_train).^p;
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
    
    acerto = sum(resultados == Y_train_validation)/double(tam_validacao)*100;
    hora = now();
    msg_resultados = sprintf('Resultados -- Acertos: %0.2f%%', acerto);
    disp(msg_resultados);
    
    arquivo_treino = sprintf('resultadosVinhoVermelho/treino-%0.2f%%-%f.mat', acerto, hora);
    arquivo_validacao = sprintf('resultadosVinhoVermelho/validacao-%0.2f%%-%f.mat', acerto, hora);
    arquivo_teste = sprintf('resultadosVinhoVermelho/teste-%0.2f%%-%f.mat', acerto, hora);
    arquivo_parametros = sprintf('resultadosVinhoVermelho/parametros-%0.2f%%-%f.mat', acerto, hora);

    dlmwrite(arquivo_treino, [X_train Y_train])
    dlmwrite(arquivo_validacao, [X_train_validation Y_train_validation])
    dlmwrite(arquivo_teste, [X_test Y_test])
    dlmwrite(arquivo_parametros, [hiper_K hiper_P num_dimensoes])

    teste_1 = find(Y_train_validation == 1);
    teste_1 = [teste_1]
    teste_2 = find(Y_train_validation == 2);
    teste_2 = [teste_2]

    resultado_1 = find(resultados == 1);
    resultado_1 = [resultado_1]
    resultado_2 = find(resultados ==  2);
    resultado_2 = [resultado_2]

    % figure();
    % plot(0,0);hold on;grid on;title(msg_resultados);
    % plot( X_test(teste_1, 1), X_test(teste_1, 2), 'bo');
    % plot( X_test(teste_2, 1), X_test(teste_2, 2), 'ro');
    % 
    % plot( X_test(resultado_1, 1), X_test(resultado_1, 2), 'b.');
    % plot( X_test(resultado_2, 1), X_test(resultado_2, 2), 'r.');
end;
