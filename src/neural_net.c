#include <cdnn/neural_net.h>
#include <cdnn/model.h>

extern __Model__ * m;

Computation_Graph * new_node(void * layer, char * type){
  Computation_Graph * new = (Computation_Graph*)malloc(sizeof(Computation_Graph));
  new->next_layer = NULL;
  new->prev_layer = NULL;
  if(!strcasecmp(type,"Dense")){
    new->DENSE = (Dense_layer*)layer;
    new->type = DENSE;
  } 
  else if(!strcasecmp(type,"Input")){
    new->INPUT = (Input_layer*)layer;
    new->type = INPUT;
  }
  else if(!strcasecmp(type,"Loss")){
    new->LOSS = (loss_layer*)layer;
    new->type = LOSS;
  }
  return new;
}

void append_graph(void * layer, char * type){
  Computation_Graph * new = new_node(layer,type);
  if(m->graph==NULL){
    m->graph = new;
    m->current_layer = new;
    m->number_of_layers+=1;
    return;
  }
  Computation_Graph * temp = m->graph;
  while(temp->next_layer!=NULL)
    temp = temp->next_layer;
  temp->next_layer = new;
  new->prev_layer = temp;
  if(strcasecmp(type,"loss")!=0)
    m->number_of_layers++;
}

void printComputation_Graph(Computation_Graph * G){
  printf("printing graph\n");
  Computation_Graph * temp = m->graph;
  while(temp!=NULL){
    if(temp->type==DENSE)
      printf("DENSE\n");
    else if(temp->type==INPUT)
      printf("INPUT\n");
    else if(temp->type==LOSS)
      printf("LOSS\n");
    else printf("NULL\n");
    temp = temp->next_layer;
  }
  return;
}

Computation_Graph * destroy_Graph(Computation_Graph * G){
  signal(SIGSEGV,segfault_handler);
  Computation_Graph * temp = m->graph;
  Computation_Graph * prev;
  while(temp!=NULL){
    prev = temp;
    temp = temp->next_layer;
    if(prev->type==DENSE){
      Dense_layer * layer = prev->DENSE;
      if(layer->weights!=NULL)
        free2d(layer->weights);
      if(layer->bias!=NULL)
        free2d(layer->bias);
      if(layer->cache!=NULL)
        free2d(layer->cache);
      if(layer->dA!=NULL)
        free2d(layer->dA);
      if(layer->A!=NULL)
        free2d(layer->A);
      if(layer->dropout_mask!=NULL)
        free2d(layer->dropout_mask);
      if(layer->db!=NULL)
        free2d(layer->db);
      if(layer->dZ!=NULL)
        free2d(layer->dZ);
      if(layer->dW!=NULL)
        free2d(layer->dW);

      layer->dW = NULL;
      layer->weights = NULL;
      layer->bias = NULL;
      layer->cache = NULL;
      layer->dA = NULL;
      layer->A = NULL;
      layer->dropout_mask = NULL;
      layer->db = NULL;
      layer->dZ = NULL;
      free(prev->DENSE);
      prev->DENSE = NULL;
      prev->prev_layer = NULL;
      layer=NULL;
    }
    else if(prev->type==INPUT){
      Input_layer * layer = prev->INPUT;
      layer->A = NULL;
      prev->prev_layer = NULL;
      free(layer);
      layer=NULL;
    }
    else if(prev->type==LOSS){
      loss_layer * layer = prev->LOSS;
      if(layer->grad_out!=NULL) free2d(layer->grad_out);
      prev->prev_layer = NULL;
      free(layer);
      layer = NULL;
    }
    prev = NULL;
  }
  prev = NULL;
  temp = NULL;
  m->graph = NULL;
  m->current_layer = NULL;
  return m->graph;
}