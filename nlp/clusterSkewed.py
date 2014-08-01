__author__="Nicholas Leonard"
__date__ ="$Oct 20, 2010 9:35:05 PM$"

import sys, time, random
sys.path.append('../')

from database import DatabaseHandler
from multi import InvisibleHand, IndependentWorker
from threading import Lock

ZERO = 0.00000000000001

ITEMLINKS_TABLE = raw_input('Enter Table name (schema.itemlinks) : ')

class Cluster:
    __doc__ = '''This class generates simple Cluster objects. 
    Each cluster object maps to a cluster_key in the PosgreSQL 
    public.itemclusters Table. The list of items hosted by the cluster 
    and their item_densities in the cluster is kept in the database only. 
    Each python cluster object is associated to this data through its 
    knowing of the cluster_key (self.key).

    Cluster objects have locks that can help the database with issues 
    related to concurrent Updates/Selects in a lazy way. Although we are 
    not sure it is actually used effectively in our code...'''
    def __init__(self, key):
        self.key = key
        self.txLock = Lock()
        self.lock = Lock()
    def get(self):
        self.lock.acquire()
        return self
    def txGet(self):
        self.txLock.acquire()
        return self

class System(InvisibleHand, DatabaseHandler):
    __doc__ = '''This class generates but one object instance. 
    It holds the state information of the whole system of clusters
    of items such as the list of clusters and online clustering information. 
    It inherits multi.InvisibleHand, a class which manages a group of workers, 
    in our case, clusteringWorkers. One of its most unique features is that it
    provides us with periodic statistics about the execution and 
    completion of the clustering system. It is also charged with the task 
    of providing clusteringWorkers with tasks, i.e. originClusters.''' 
    def __init__(self):
		start = time.time()
		DatabaseHandler.__init__(self)
		#init SQL Functions:
		self.executeSQL('''
         CREATE OR REPLACE FUNCTION public.measure_density(item_key INT4, cluster_key INT4)
            RETURNS FLOAT8 AS $$
         SELECT GREATEST(SUM(similarity), 0.0000000000000001) AS sum
         FROM %s AS a, public.itemclusters AS b
         WHERE $1 = a.tail AND a.head = b.item_key AND b.cluster_key = $2
         $$ LANGUAGE 'SQL';
      ''' % (ITEMLINKS_TABLE,))
		self.executeSQL('''
        CREATE OR REPLACE FUNCTION public.get_clustering_statistics()
                RETURNS TABLE(count INT4, maxsum FLOAT8, maxcount INT2, sum FLOAT8) AS $$
        SELECT COUNT(*)::INT4, MAX(sum), MAX(count)::INT2, SUM(sum)
        FROM    (
                SELECT cluster_key, COUNT(*), SUM(density)
                FROM public.itemclusters
                GROUP BY cluster_key
                ORDER BY sum DESC
                ) AS foo
        WHERE sum > 50
        ; $$ LANGUAGE 'SQL';
		''')
		self.executeSQL('''
        CREATE OR REPLACE FUNCTION public.measure_density_for_transfer(affected_item INT4, transfered_item INT4)
            RETURNS FLOAT8 AS $$
        SELECT GREATEST(SUM(similarity), 0.000001) AS sum
		FROM 	(
			SELECT cluster_key
			FROM public.itemclusters
			WHERE item_key = $1
			) AS a, %s AS b, public.itemclusters AS c
		WHERE $1 = b.tail AND b.head = c.item_key AND c.item_key != $2 AND c.cluster_key = a.cluster_key
		; $$ LANGUAGE 'SQL';
		''' % (ITEMLINKS_TABLE,))
		self.executeSQL('''
        CREATE OR REPLACE FUNCTION public.transfer_item_to_cluster(item INT4, new_cluster INT4)
            RETURNS VOID  AS $$
        --Update density of all items in previous_cluster itemlinking the item:
        UPDATE public.itemclusters
        SET density = public.measure_density_for_transfer(item_key, $1)
        FROM 	(
                SELECT cluster_key AS previous_cluster
                FROM public.itemclusters
                WHERE item_key = $1
                ) AS a, 
                (
				SELECT head AS item
				FROM %s
				WHERE $1 = tail
                ) AS b
        WHERE b.item = item_key AND cluster_key = a.previous_cluster;
        --transfer the item to new_cluster:
        UPDATE public.itemclusters
        SET cluster_key = $2, density = public.measure_density($1, $2)
        WHERE item_key = $1;
        --Update item_density of all items in the new_cluster itemlinking the item:
        UPDATE public.itemclusters
        SET density = public.measure_density(item_key, $2)
        FROM 	(
				SELECT head AS item
				FROM %s
				WHERE $1 = tail
				) AS a
        WHERE a.item = item_key AND cluster_key = $2
        ; $$ LANGUAGE 'SQL';
		''' % (ITEMLINKS_TABLE, ITEMLINKS_TABLE))
		self.executeSQL('''
        CREATE OR REPLACE FUNCTION public.get_badly_clustered_item_from_cluster(cluster INT4)
            RETURNS INT4 AS $$
        SELECT item_key
        FROM public.itemclusters
        WHERE cluster_key = $1
        ORDER BY density ASC 
        LIMIT 1
        ; $$ LANGUAGE 'SQL';
        ''')
        #get clusters:
		clusters = self.executeSQL("SELECT DISTINCT cluster_key FROM public.itemclusters", action=self.FETCH_ALL)
		self.clusters = {}
		self.clusterList = []
		for (cluster_key,) in clusters:
			self.clusters[cluster_key] = Cluster(cluster_key)
			self.clusterList.append(cluster_key)
		self.num_clusters = len(self.clusters)
		self.taskLock = Lock()
		self.nextTask = 1
		#for real time (periodic online) statistics:
		self.taskings = ZERO; self.changes = ZERO; self.tries = ZERO;
		self.loopFailures = 0;
		self.lastTimeCheck = time.time()
		self.lastSync = time.time()
		#init workers:
		InvisibleHand.__init__(self, ClusteringWorker)
		print "system initialized in", time.time() - start, "secs"
    def clusterExists(self, cluster_key):
        return cluster_key in self.clusters
    def getTask(self):
        return random.choice(self.clusterList)
    def _reportSituation(self):
        changes = ZERO; tries = ZERO; loopFailures = 0; taskings = ZERO
        for worker in self.workers:
            (c, i, lf) = worker.getAndResetChanges()
            changes += c; tries += i; loopFailures += lf;
            taskings += worker.getAndResetCompletedTasks()
        self.taskings += taskings; self.tries += tries;
        self.loopFailures += loopFailures; self.changes += changes;
        taskings = float(max(taskings, ZERO))
        lastTimeCheck = time.time()
        elapsed = float(lastTimeCheck - self.lastTimeCheck)
        self.lastTimeCheck = lastTimeCheck
        clusterInfo = self.executeSQL('SELECT * FROM public.get_clustering_statistics()', action = self.FETCH_ONE)
        print "We have", len(self.clusters), "clusters"
        print "--Speeds:"
        print taskings/elapsed, "t/sec,", changes/elapsed, "changes/sec;", tries/elapsed, "i/sec"
        print "--Counters"
        print self.taskings, "Taskings"
        print self.loopFailures, "LoopFailures"
        print self.tries, "Iterations"
        print self.changes, "Changes"
        print clusterInfo
        print "--Ratios"
        print changes/taskings, "changes/t", self.changes/self.taskings, "Changes/T"
        print changes/tries, "changes/i;", self.changes/self.tries, "Changes/I"
        print self.tries/self.taskings, "I/T"
        print self.loopFailures/self.taskings, "LoopFailures/T"
        print "-------------------------------------------------"

class ClusteringWorker(IndependentWorker, DatabaseHandler):
    __doc__ = '''This class generates clustering workers (threads) of 
    the form multi.IndependentWorker. Each such worker manages its own 
    database connection. Hence, each clusteringWorker python thread is 
    linked to a PostgreSQL process. Because of python's implemention 
    of a global interpreter lock (GIL), the best way to make use of
    all our CPU cores is to offload as much work to PostgreSQL 
    postmasters (processes). ClusteringWorker objects are responsible 
    with the highlevel clustering logic which determines when to query 
    the database and when to make changes. All SQL queries where 
    refactored into SQL functions which speeds things up and makes 
    the code more readable.

    A clusteringWorker receives a cluster_key as a tasking from the 
    System object when it does not have work. This cluster_key is the 
    originCluster from which this process starts. It starts off by picking 
    some items from the originCluster. For each such item, it emits an 
    SQL Select to determine which clusters exert the greatest pull on them. 
    If a subject item is found to be pulled by a cluster whose item with 
    the lowest item_density has less item_density than the item_density 
    the subject item is predicted to have as a member of that cluster, 
    then that subject item is transfered to that cluster. 
    And the clusteringWorker moves to that cluster to find another item
    in the same way. If the cluster which the clusteringWorker moves to 
    is the originCluster, the tasking loop is closed and the 
    clusteringWorkers asks the System for another originCluster.
    The principle implemented by this logic aims to keep all clusters 
    with an equal amount of items while making wise item transfers. 
    If a clusteringWorker fails to find an item to transfer, then it 
    chooses the item with the lowest density and transfers it to the
    originCluster in order to close the item exchange loop for this task.'''
    def __init__(self, id, system, workerArgs = None):
        DatabaseHandler.__init__(self)
        self.changesLock = Lock()
        #this is incremented for SQL Updates used to transfer items from cluster to cluster.
        self.changes = 0
        #this is incremented for each SQL Select emited to determine where an item is most pulled. 
        self.i = 0
        self.itemExchangeLoopClosingFailures = 0 
        IndependentWorker.__init__(self, id, system)
    def _task(self, cluster_key):
        self.taskPath = []
        self.visitedClusters = {}
        self.taskPath.append(cluster_key)
        self.sbjCluster = self.manager.clusters[cluster_key].get()
        nextSbjCluster = self.findNextSbjCluster()
        while nextSbjCluster and (nextSbjCluster != self.taskPath[0]):
            #print "balanced sbjCluster; unbalanced nextSbjCluster;"
            #print "item exchange loop is still open; continue tasking"
            if self.sbjCluster.key != self.taskPath[0]:
                if nextSbjCluster in self.visitedClusters:
                    #remove the closed loop of visited clusters:
                    self.taskPath = self.taskPath[:self.visitedClusters[nextSbjCluster]]
                    del self.visitedClusters[nextSbjCluster]
                #add our current sbjCluster to the map of visited clusters and list of our path:
                self.visitedClusters[self.sbjCluster.key] = len(self.visitedClusters)+1
                self.taskPath.append(self.sbjCluster.key)
            self.sbjCluster = self.manager.clusters[nextSbjCluster].get()
            nextSbjCluster = self.findNextSbjCluster()
        #print "item exchange loop is closed; this tasking is done"
        return True
    def findNextSbjCluster(self):
        find_i = 0
        for (sbjItem,) in self.executeSQL('''
            SELECT item_key
				FROM	(
                     (
                     SELECT item_key, density
                     FROM public.itemclusters
                     WHERE cluster_key = %s
                     OFFSET (random()*7)::INT4
                     LIMIT 3
                     )
                     UNION
                     (
                     SELECT item_key, density
                     FROM public.itemclusters
                     WHERE cluster_key = %s
                     ORDER BY density ASC
                     LIMIT 3
                     ) 
                  ) AS a
				ORDER BY density ASC
            ''', (self.sbjCluster.key, self.sbjCluster.key), self.FETCH_ALL):
            nextSbjCluster = self.findBetterHostClusterForItem(sbjItem)
            find_i += 1
            if nextSbjCluster:
                if len(self.visitedClusters) > 100:
                    #print "taskPath is too great; close the loop", self.id, nextSbjCluster, find_i, sbjItem
                    break
                #print "transfering the item to nextSbjCluster"
                receivingCluster = self.manager.clusters[nextSbjCluster].txGet()
                if not self.executeSQL("SELECT public.transfer_item_to_cluster(%s, %s)", (sbjItem, nextSbjCluster)):
                    if not self.executeSQL("SELECT public.transfer_item_to_cluster(%s, %s)", (sbjItem, nextSbjCluster)):
                        receivingCluster.txLock.release()
                        continue
                receivingCluster.txLock.release()
                self.changesLock.acquire()
                self.changes += 1
                self.i += find_i
                self.changesLock.release()
                self.sbjCluster.lock.release()
                return nextSbjCluster
        #print "wasn't able to continue or close the tasking loop;"
        #print "transfering badly clustering item in sbjCluster to originCluster; closes the item exchange loop"
        receivingCluster = self.manager.clusters[self.taskPath[0]].txGet()
        if not self.executeSQL("SELECT public.transfer_item_to_cluster(public.get_badly_clustered_item_from_cluster(%s), %s)", (self.sbjCluster.key, self.taskPath[0])):
            time.sleep(1)
            if not self.executeSQL("SELECT public.transfer_item_to_cluster(public.get_badly_clustered_item_from_cluster(%s), %s)", (self.sbjCluster.key, self.taskPath[0])):
                time.sleep(1)
                self.executeSQL("SELECT public.transfer_item_to_cluster(public.get_badly_clustered_item_from_cluster(%s), %s)", (self.sbjCluster.key, self.taskPath[0]))
        receivingCluster.txLock.release()
        self.changesLock.acquire()
        self.itemExchangeLoopClosingFailures += 1
        self.i += find_i
        self.changesLock.release()
        self.sbjCluster.lock.release()
        return False
    def findBetterHostClusterForItem(self, sbjItem):
        mostInfluentialClusters = self.executeSQL('''
        SELECT cluster_key
        FROM(
            SELECT cluster_key, SUM(similarity*density) AS sd_pull, SUM(similarity) AS potentialself_density
            FROM public.itemclusters, %s
            WHERE %s = tail AND head = item_key
            GROUP BY cluster_key
            ORDER BY sd_pull DESC
            ) AS a,
            (
            SELECT density
            FROM public.itemclusters
            WHERE item_key = %s
            ) AS b
        WHERE(
               potentialself_density >= 
               (
               SELECT MIN(density)*1.1::FLOAT8
               FROM public.itemclusters AS b
               WHERE a.cluster_key = b.cluster_key
               )
            ) 
        AND (potentialself_density > b.density)
        ''' % (ITEMLINKS_TABLE, sbjItem, sbjItem), action = self.FETCH_ALL)
        for (objClusterKey,) in mostInfluentialClusters:
            if objClusterKey == self.sbjCluster.key:
                #print "the item is well clustered"
                return False
            else:
                #print "the item will be transfered"
                return objClusterKey
        return False
    def getAndResetChanges(self):
        changes = self.changes
        i = self.i
        itemExchangeLoopClosingFailures = self.itemExchangeLoopClosingFailures
        self.changesLock.acquire()
        self.changes = 0
        self.i = 0
        self.itemExchangeLoopClosingFailures = 0
        self.changesLock.release()
        return (changes, i, itemExchangeLoopClosingFailures)
    def _close(self):
        self.conn.close()


if __name__ == '__main__':
    s = System()
    s.lead()
