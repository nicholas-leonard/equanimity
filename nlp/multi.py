__author__="Nicholas Léonard"
__date__ ="$Oct 20, 2010 9:35:05 PM$"

import sys, time
from threading import Thread, Lock

NUM_WORKERS = 8
CHECK_INTERVAL = 7

#Independent workers model (no queue):
'''
While using a queue involves the manager into pushing jobs, here it
will be the workers that will ask for jobs. We hope to reduce management
overhead.
'''

SLEEP = 2 #mins
SYNC = 1 #hours

class InvisibleHand:
    def __init__(self, worker, workerArgs = None, syncs = []):
        self.numWorkers = int(raw_input("Number of workers or cores: "))
        sys.setcheckinterval(CHECK_INTERVAL)
        self.worker = worker
        self.workerArgs = workerArgs
        self.sleep = SLEEP*60
        self.syncEverySeconds = SYNC*60*60
        self.syncs = syncs
        self.syncRequest = False
    def _startWorking(self):
        "initializes the workers and tells them to start working"
        self.workers = [self.worker( i, self, self.workerArgs) for i in range(self.numWorkers)]
        for worker in self.workers:
            worker.start()
        print "workers working"
    def _observe(self):
        self.start = time.time()
        while True:
            self._reportSituation()
            time.sleep(self.sleep)
            if time.time() - self.start > self.syncEverySeconds:
                self.syncRequest = True
                self._sync()
                self.start = time.time()
    def _reportSituation(self):
        '''Fill this method with code that can print you something'''
        pass
    def _sync(self):
        pass
    def lead(self):
        '''This is init function'''
        self._startWorking()
        self._observe()
        self._daysWork()
        sys.exit()
    def _daysWork(self):
        '''Call it a day. Close the shop, and join your Threaded workers.'''
        print "joining"
        for worker in self.workers:
            worker.join()
        self._close()
        print "joined and Done"
    def getTask(self):
        '''Replace this function with one that returns a
        task when asked by a worker.'''
        pass
    def _close(self):
        '''replace with code that needs to be executed after all threads
        have been joined during a _daysWork()'''
        pass

class IndependentWorker(Thread):
    def __init__(self, id, manager, workerArgs = None):
        '''A worker needs a task stack(queue), a name (id) and a boss (manager)'''
        self.id = id
        self.manager = manager
        self.completedTasksLock = Lock()
        self.completedTasks = 0
        Thread.__init__(self)
    def run(self):
        subject = self.manager.getTask()
        while subject != None:
            if self._task(subject):
                self.completedTasksLock.acquire()
                self.completedTasks += 1
                self.completedTasksLock.release()
            subject = self.manager.getTask()
        self._close()
        sys.exit()
    def getAndResetCompletedTasks(self):
        completedTasks = self.completedTasks
        self.completedTasksLock.acquire()
        self.completedTasks = 0
        self.completedTasksLock.release()
        return completedTasks
    def _task(self, subject):
        print 'You should initialize Worker._task(self, subject)'
        sys.exit()
        pass
    def _close(self):
        pass


