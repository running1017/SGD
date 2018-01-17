function [w1, w2] = nag(w1_0, w2_0, x, y, gradw1, gradw2, eta, gamma, batch, index)
    [epoch, datanum] = size(index);
    slice = ceil(datanum/batch);
    w1 = [w1_0, zeros(1, epoch*slice)];
    w2 = [w2_0, zeros(1, epoch*slice)];
    dw1 = 0; dw2 = 0;

    for i = 0:epoch-1
        for j = 1:slice
            t = (j*batch<=datanum)*(j*batch) + (j*batch>datanum)*datanum;
            dw1 = gamma*dw1 + eta(i*slice+j-1)*...
                gradw1(w1(i*slice+j) - gamma*dw1, w2(i*slice+j) - gamma*dw2, x((j-1)*batch+1:t), y((j-1)*batch+1:t));
            dw2 = gamma*dw2 + eta(i*slice+j-1)*...
                gradw2(w1(i*slice+j) - gamma*dw1, w2(i*slice+j) - gamma*dw2, x((j-1)*batch+1:t), y((j-1)*batch+1:t));
            w1(i*slice+j+1) = w1(i*slice+j) - dw1;
            w2(i*slice+j+1) = w2(i*slice+j) - dw2;
        end
    end
end