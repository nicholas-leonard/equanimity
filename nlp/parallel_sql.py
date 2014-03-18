#!/usr/bin/env python

from database import DatabaseHandler
import time, sys
from threading import Thread, Lock

class ParallelQueryWorker(DatabaseHandler, Thread):
    #Replace this with the query that will be executed to fetch a list of Items:
    GetItemsQuery = None #ex: "SELECT idea_key FROM tanimoto.en_idea_weights2"
    #Replace this with query string that will be executed in parallel for each Item:
    ParallelQuery = None #ex: "SELECT tanimoto.fix_tanimoto_error_and_join(%s)"
    #Set the number of item iterations between each time.sleep(10) if your
    # ParallelQuery creates lots of temp Tables, else leave it as such:
    WaitEveryIterations = None
    #Set this to the number of cores you have.
    # If your have more than 5-8, keep one core out of the equation:
    NumCores = 7
    def __init__(self, id="main", master=None):
        DatabaseHandler.__init__(self)
        Thread.__init__(self)
        self._id = id
        if master is None :
            self._itemList = self._getItems()
            self._items = len(self._itemList)
            self._itemsParsedLock = Lock()
            self._itemsParsed = 0
            self.main = self
            self._power = True
            self._sync = False
            self._report = False
        else:
            self.main = master
            self._power = False
    def _startWorking(self):
        "Initializes the workers and tells them to start working"
        self.workers = [ParallelQueryWorker(i, self)
                        for i in range(ParallelQueryWorker.NumCores)]
        for worker in self.workers:
            worker.start()
        print "workers working"
    def _daysWork(self):
        '''Call it a day. Close the shop, and join your Threaded workers.'''
        print "joining"
        for worker in self.workers:
            worker.join() 
        print "joined and Done"
    def run(self):
        if self._power:
            self._startWorking()
            start = time.time()
        nextItem = 0
        while nextItem < self.main._items:
            self.main._itemsParsedLock.acquire()
            nextItem = self.main._itemsParsed
            self.main._itemsParsed+=1
            self.main._itemsParsedLock.release()
            if nextItem >= self.main._items:
                break
            self.executeSQL(ParallelQueryWorker.ParallelQuery, self.main._itemList[nextItem])
            if nextItem % 500 == 0 and not self.main._report:
                self.main._report = True
            if self._power and self.main._report:
                print self._itemsParsed, "items parsed of", self._items
                period = time.time() - start
                cost = period/self._itemsParsed #seconds / item
                print self._itemsParsed/period, "items/sec", cost , "sec/item"
                print cost*(self._items-self._itemsParsed)/60, "mins left"
		self._report=False
        if self._power:
            self._daysWork()
            print "done (really)"
    def _getItems(self):
        print "pql is RAM-memorizing its items..."
        start = time.time()
        items = self.executeSQL(ParallelQueryWorker.GetItemsQuery, action = self.FETCH_ALL)
        print "pql RAM-memorized its %s items in %d seconds" % \
            (len(items), int(time.time() - start))
        return items

if __name__ == '__main__':
    if (len(sys.argv) != 3) and (len(sys.argv) != 4):
        print '''parallel_sql "SELECT idea_key FROM wiki.fr_ideas" "SELECT tanimoto.fr_insert_ring2_tail(%s)" [num_cores]'''
        sys.exit()
    if len(sys.argv) == 4:
    	ParallelQueryWorker.NumCores = int(sys.argv[3])-1
    ParallelQueryWorker.ParallelQuery = sys.argv[2] #"SELECT tanimoto.fr_insert_ring2_tail(%s)"
    ParallelQueryWorker.GetItemsQuery = sys.argv[1] #"SELECT idea_key FROM wiki.fr_ideas"
    pqw = ParallelQueryWorker()
    pqw.run()
