#pragma once
#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

// Only a guaranteed lock-free atomic operation runs inside the signal handler.
inline std::atomic<bool> pause_signal{false};
static_assert(std::atomic<bool>::is_always_lock_free);
inline void request_pause(int) { pause_signal.store(true, std::memory_order_relaxed); }

class PauseSignals {
    using Handler = void (*)(int);
    Handler old_int_, old_term_;
public:
    PauseSignals() {
        pause_signal.store(false);
        old_int_=std::signal(SIGINT, request_pause);
        if(old_int_==SIG_ERR) throw std::runtime_error("Cannot install SIGINT handler");
        old_term_=std::signal(SIGTERM, request_pause);
        if(old_term_==SIG_ERR) {
            std::signal(SIGINT,old_int_);
            throw std::runtime_error("Cannot install SIGTERM handler");
        }
    }
    ~PauseSignals() { std::signal(SIGINT, old_int_); std::signal(SIGTERM, old_term_); }
    PauseSignals(const PauseSignals&) = delete;
};

// Advisory/deny-sharing locks are released by the OS even after a forced kill.
// The persistent lock file is harmless; there is no stale-directory lock to clear.
class RunLock {
    int fd_ = -1;
public:
    explicit RunLock(const std::filesystem::path& directory) {
        const auto path = (directory / ".run.lock").string();
#ifdef _WIN32
        if (_sopen_s(&fd_, path.c_str(), _O_CREAT|_O_RDWR|_O_BINARY,
                     _SH_DENYRW, _S_IREAD|_S_IWRITE) != 0)
            throw std::runtime_error("Output directory is locked or inaccessible: " + path);
#else
        fd_ = ::open(path.c_str(), O_CREAT|O_RDWR, 0600);
        if (fd_ < 0) throw std::runtime_error("Cannot open run lock: " + path);
        if (::flock(fd_, LOCK_EX|LOCK_NB) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("Another process is using the output directory: " + path);
        }
#endif
    }
    ~RunLock() {
        if (fd_ < 0) return;
#ifdef _WIN32
        _close(fd_);
#else
        ::flock(fd_, LOCK_UN); ::close(fd_);
#endif
    }
    RunLock(const RunLock&) = delete;
};

inline void durable_flush(FILE* file) {
    if (std::fflush(file) != 0) throw std::runtime_error("Journal flush failed");
#ifdef _WIN32
    if (_commit(_fileno(file)) != 0) throw std::runtime_error("Journal disk sync failed");
#else
    if (::fsync(::fileno(file)) != 0) throw std::runtime_error("Journal disk sync failed");
#endif
}
