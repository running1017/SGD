function [w1, w2] = adadelta(w1_0, w2_0, x, y, gradw1, gradw2, gamma, vareps, batch, index)
    [epoch, datanum] = size(index);
    slice = ceil(datanum/batch);
    w1 = [w1_0, zeros(1, epoch*slice)];
    w2 = [w2_0, zeros(1, epoch*slice)];
    sw1 = vareps; sw2 = vareps;
    gw1 = vareps; gw2 = vareps;
    
    for i = 0:epoch-1
        for j = 1:slice
            t = (j*batch<=datanum)*(j*batch) + (j*batch>datanum)*datanum;
            gw1 = gamma*gw1 + (1-gamma)*gradw1(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t))^2;
            gw2 = gamma*gw2 + (1-gamma)*gradw2(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t))^2;
            dw1 = sqrt(sw1+vareps) / sqrt(gw1+vareps) * gradw1(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t));
            dw2 = sqrt(sw2+vareps) / sqrt(gw2+vareps) * gradw2(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t));
            sw1 = gamma*sw1 + (1-gamma)*dw1^2;
            sw2 = gamma*sw2 + (1-gamma)*dw2^2;
            w1(i*slice+j+1) = w1(i*slice+j) - dw1;
            w2(i*slice+j+1) = w2(i*slice+j) - dw2;
        end
    end
end