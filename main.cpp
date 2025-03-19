#include "Environment/Environment.h"

int main() {
    Environment env = Environment(4, 7, 10, 123, true);

    env.step(0);
    env.step(1);
    env.step(2);
    env.step(3);

    return 0;
}