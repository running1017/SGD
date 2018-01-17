function [w1, w2] = adagrad(w1_0, w2_0, x, y, gradw1, gradw2, eta, vareps, batch, index)
    [epoch, datanum] = size(index);
    slice = ceil(datanum/batch);
    w1 = [w1_0, zeros(1, epoch*slice)];
    w2 = [w2_0, zeros(1, epoch*slice)];
    gw1 = vareps; gw2 = vareps;
    
    for i = 0:epoch-1
        for j = 1:slice
            t = (j*batch<=datanum)*(j*batch) + (j*batch>datanum)*datanum;
            dw1 = gradw1(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t));
            dw2 = gradw2(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t));
            gw1 = gw1 + dw1*dw1;
            gw2 = gw2 + dw2*dw2;
            w1(i*slice+j+1) = w1(i*slice+j) - eta(i*slice+j) / sqrt(gw1) * dw1;
            w2(i*slice+j+1) = w2(i*slice+j) - eta(i*slice+j) / sqrt(gw2) * dw2;
        end
    end
end