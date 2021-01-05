#include "../layers/Dense.h"

enum layer_type {NONE, INPUT, DENSE, LOSS, OPTIMIZER};

typedef struct computational_graph{
  struct computational_graph * next_layer;
  struct computational_graph * prev_layer;
  enum layer_type type;
  union
  {
    Dense_layer * DENSE;
  };
}Computation_Graph;

Computation_Graph * G;

Computation_Graph * init(){
  Computation_Graph * G = (Computation_Graph*)malloc(sizeof(Computation_Graph));
  G->next_layer = NULL;
  G->prev_layer = NULL;
  G->type = NONE;
  G->DENSE = NULL;
  return G;
}

Computation_Graph * new_node(void * layer, char * type){
  Computation_Graph * G = (Computation_Graph*)malloc(sizeof(Computation_Graph));
  G->next_layer = G->prev_layer = NULL;
  if(!strcmp(type,"Dense")){
    G->DENSE = (Dense_layer*)layer;
    G->type = DENSE;
  } 
  return G;
}

Computation_Graph * append_graph(Computation_Graph * G, void * layer, char * type){
  Computation_Graph * new = new_node(layer,type);
  if(G==NULL){
    G = new;
    return G;
  }
  Computation_Graph * temp = G;
  while(temp->next_layer!=NULL){
    temp = temp->next_layer;
  }
  //now we are on the last node
  temp->next_layer = new;
  new->prev_layer = temp;
  return G;
}

void printComputation_Graph(Computation_Graph * G){
  Computation_Graph * temp = G;
  while(temp!=NULL){
    if(temp->type==DENSE)
      printf("DENSE\n");
    else printf("NULL\n");
    temp = temp->next_layer;
  }
  return;
}

Computation_Graph * destroy_G(Computation_Graph * G){
  Computation_Graph * temp = G;
  Computation_Graph * prev;
  while(temp!=NULL){
    prev = temp;
    temp = temp->next_layer;
    free(prev->DENSE);
    free(prev);
  }
  prev = NULL;
  temp = NULL;
  G = NULL;
  return G;
}
