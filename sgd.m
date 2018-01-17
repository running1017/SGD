function [w1, w2] = sgd(w1_0, w2_0, x, y, gradw1, gradw2, eta, batch, index)
    [epoch, datanum] = size(index);
    slice = ceil(datanum/batch);
    w1 = [w1_0, zeros(1, epoch*slice)];
    w2 = [w2_0, zeros(1, epoch*slice)];
    
    for i = 0:epoch-1
        for j = 1:slice
            t = (j*batch<=datanum)*(j*batch) + (j*batch>datanum)*datanum;
            w1(i*slice+j+1) = w1(i*slice+j) -...
                eta(i*slice+j)*gradw1(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t));
            w2(i*slice+j+1) = w2(i*slice+j) -...
                eta(i*slice+j)*gradw2(w1(i*slice+j), w2(i*slice+j), x((j-1)*batch+1:t), y((j-1)*batch+1:t));
        end
    end
end