:auto MATCH ()-[r]->() 
CALL { WITH r 
DELETE r 
} IN TRANSACTIONS OF 50000 ROWS;

:auto MATCH (n) 
CALL { WITH n 
DETACH DELETE n 
} IN TRANSACTIONS OF 50000 ROWS;