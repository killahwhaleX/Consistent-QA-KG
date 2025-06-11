MATCH ()-[r]->() 
CALL (r) { WITH r 
DELETE r 
} 
