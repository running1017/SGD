function [w1, w2] = adam(w1_0, w2_0, x, y, gradw1, gradw2, eta, beta1, beta2, vareps, batch, index)
    [epoch, datanum] = size(index);
    slice = ceil(datanum/batch);
    w1 = [w1_0, zeros(1, epoch*slice)];
    w2 = [w2_0, zeros(1, epoch*slice)];
    mw1 = 0; mw2 = 0;
    vw1 = 0; vw2 = 0;
    
    for i = 0:epoch-1
        for j = 1:slice
            t = (j*batch<=datanum)*(j*batch) + (j*batch>datanum)*datanum;
            mw1 = beta1*mw1 + (1-beta1)* ...
                gradw1(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t));
            mw2 = beta1*mw2 + (1-beta1)* ...
                gradw2(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t));
            vw1 = beta2*vw1 + (1-beta2)* ...
                gradw1(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t))^2;
            vw2 = beta2*vw2 + (1-beta2)* ...
                gradw2(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t))^2;
            w1(i*slice+j+1) = w1(i*slice+j) - eta(i*slice+j)*mw1/(sqrt(vw1)+vareps);
            w2(i*slice+j+1) = w2(i*slice+j) - eta(i*slice+j)*mw2/(sqrt(vw2)+vareps);
        end
    end
end