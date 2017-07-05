load('q_reward_env1')
load('gpq_reward_env1')
load('deep_q_reward')

[n1,p1] = size(q_reward_env1);
[n2,p2] = size(gpq_reward_env1);
[n3,p3] = size(deep_q_reward);

t1 = 1:n1;
t2 = 1:n2;
t3 = 1:n3;


plot(t1,q_reward_env1);
hold on
plot(t2,gpq_reward_env1);
hold on
plot(t3,deep_q_reward);

