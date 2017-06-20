load('q_reward_env4')
load('gpq_reward_env4')
load('gpq_reward_env4_2')

[n1,p1] = size(q_reward_env4);
[n2,p2] = size(gpq_reward_env4);
[n3,p3] = size(gpq_reward_env4_2);

t1 = 1:n1;
t2 = 1:n2;
t3 = 1:n3;

plot(t1,q_reward_env4);
hold on
%plot(t2,gpq_reward_env4);
%hold on;
plot(t3,gpq_reward_env4_2);
