#include "../layers/layers.h"

enum layer_type {NONE, INPUT, DENSE, LOSS, OPTIMIZER};

typedef struct computational_graph{
  struct computational_graph * next_layer;
  struct computational_graph * prev_layer;
  enum layer_type type;
  union
  {
    Dense_layer * DENSE;
    Input_layer * INPUT;
  };
}Computation_Graph;

Computation_Graph * G;

int computation_graph_status;//0 - Fprop 1- Backprop
char * loss_type;
dARRAY * Y;

Computation_Graph * init(){
  Computation_Graph * G = (Computation_Graph*)malloc(sizeof(Computation_Graph));
  G->next_layer = NULL;
  G->prev_layer = NULL;
  G->type = NONE;
  G->DENSE = NULL;
  G->INPUT = NULL;
  return G;
}

Computation_Graph * new_node(void * layer, char * type){
  Computation_Graph * G = (Computation_Graph*)malloc(sizeof(Computation_Graph));
  G->next_layer = G->prev_layer = NULL;
  if(!strcmp(type,"Dense")){
    G->DENSE = (Dense_layer*)layer;
    G->type = DENSE;
  } 
  else if(!strcmp(type,"Input")){
    printf("Appending input layer!\n");
    G->INPUT = (Input_layer*)layer;
    G->type = INPUT;
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
    else if(temp->type==INPUT)
      printf("INPUT\n");
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
    if(prev->type==DENSE){
      Dense_layer * layer = prev->DENSE;
      free(layer->weights);
      layer->weights = NULL;
      free(layer->bias);
      layer->bias = NULL;
      free(layer->cache);
      layer->cache = NULL;
      free(layer->dA);
      layer->dA = NULL;
      free(layer->A);
      layer->A = NULL;
      free(layer->dropout_mask);
      layer->dropout_mask = NULL;
      free(layer->db);
      layer->db = NULL;
      free(layer->dZ);
      layer->dZ = NULL;
      free(layer->dW);
      layer->dW = NULL;
      free(prev->DENSE);
      prev->DENSE = NULL;
      prev->prev_layer = NULL;
      layer=NULL;
    }
    else if(prev->type==INPUT){
      Input_layer * layer = prev->INPUT;
      free(layer->A);
      layer->A = NULL;
      prev->prev_layer = NULL;
      free(prev->INPUT);
      layer=NULL;
    }
    free(prev);
  }
  prev = NULL;
  temp = NULL;
  G = NULL;
  return G;
}
double cross_entropy_loss(Dense_layer * layer, dARRAY * Y);
