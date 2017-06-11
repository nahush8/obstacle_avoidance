load('q_reward')
load('gpq_reward')
load('deep_q_reward')

[n1,p1] = size(q_reward);
[n2,p2] = size(gpq_reward);
[n3,p3] = size(deep_q_reward);

t1 = 1:n1;
t2 = 1:n2;
t3 = 1:n3;

plot(t1,q_reward);
hold on
plot(t2,gpq_reward);
hold on
plot(t3,deep_q_reward);
