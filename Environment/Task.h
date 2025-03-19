#ifndef TASK_H
#define TASK_H

class Task {
public:
    explicit Task(unsigned length);
    void tick();
    bool isFinished() const;
    unsigned getRemainingDuration() const;
private:
    unsigned _duration;
};

#endif //TASK_H
