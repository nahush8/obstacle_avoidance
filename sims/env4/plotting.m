load('q_reward_env4')
load('gpq_reward_env4')


[n1,p1] = size(q_reward_env4);
[n2,p2] = size(gpq_reward_env4);

t1 = 1:n1;
t2 = 1:n2;


plot(t1,q_reward_env4);
hold on
plot(t2,gpq_reward_env4);

