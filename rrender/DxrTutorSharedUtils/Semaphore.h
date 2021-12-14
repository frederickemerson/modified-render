#include <mutex>
#include <condition_variable>

// Helper class for a simple binary semaphore
// Implemented using a mutex and a conditional variable
class Semaphore {
public:
    // Initialises a semaphore with an initial value
    // If true : Semaphore is initally unlocked
    //           First wait() will not block
    // If false: Semaphore is initially locked
    //           First wait() will block until the first signal()
    Semaphore(bool initialValue);
    
    void signal();
    void wait();

private:
    std::mutex mutex;
    std::condition_variable condVar;
    bool isReady;
};
