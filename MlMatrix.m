function [Dl, Ml] = MlMatrix(TrainMatrix,Labels)

Wr=zeros(size(TrainMatrix,2));
We=zeros(size(TrainMatrix,2));
for i=1:size(TrainMatrix,2)
    for j=1:size(TrainMatrix,2)
        if Labels(i) == Labels(j)
            We(i,j) = 0;
            Wr(i,j) = 1/sum(Labels == Labels(i));
        else
            We(i,j) = 1/sum(Labels ~= Labels(i));
            Wr(i,j) = 0;
        end
    end
end
Ml = 3 * eye(size(TrainMatrix,2)) + diag(sum(We)) + We + We' - 2 * Wr;
Dl = eye(size(TrainMatrix,2)) + diag(sum(We));
end