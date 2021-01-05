#include "../layers/Dense.h"

enum layer_type {NONE, DENSE, LOSS, OPTIMIZER};

typedef struct computational_graph{
  struct computational_graph * next;
  struct computational_graph * prev;
  enum layer_type type;
  union
  {
    Dense_layer * DENSE;
  };
}Computation_Graph;

Computation_Graph * G;

Computation_Graph * init(){
  Computation_Graph * G = (Computation_Graph*)malloc(sizeof(Computation_Graph));
  G->next = NULL;
  G->prev = NULL;
  G->type = NONE;
  G->DENSE = NULL;
  return G;
}

Computation_Graph * new_node(void * layer, char * type){
  printf("I created a new G for the layer\n");
  Computation_Graph * G = (Computation_Graph*)malloc(sizeof(Computation_Graph));
  G->next = G->prev = NULL;
  printf("Checking for type of layer\n");
  if(!strcmp(type,"Dense")){
    G->DENSE = (Dense_layer*)layer;
    G->type = DENSE;
  } 
  printf("Returning G\n");
  return G;
}

Computation_Graph * append_graph(Computation_Graph * G, void * layer, char * type){
  printf("I came to append layer\n");
  Computation_Graph * new = new_node(layer,type);
  if(G==NULL){
    printf("Computation_Graph was null\n");
    G = new;
    return G;
  }
  G->next = new;
  printf("I have linked the layer\n");
  new->prev = G;
  printf("Returning!\n");
  return G;
}

void printComputation_Graph(Computation_Graph * G){
  printf("Assigned a temp pointer to G\n");
  Computation_Graph * temp = G;
  printf("Printing\n");
  while(temp!=NULL){
    if(temp->type==DENSE)
      printf("DENSE\n");
    else printf("NULL\n");
    temp = temp->next;
  }
  return;
}

Computation_Graph * destroy_G(Computation_Graph * G){
  Computation_Graph * temp = G;
  Computation_Graph * prev;
  while(temp!=NULL){
    prev = temp;
    temp = temp->next;
    free(prev->DENSE);
    free(prev);
  }
  prev = NULL;
  temp = NULL;
  G = NULL;
  return G;
}
