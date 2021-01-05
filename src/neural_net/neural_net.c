// #include "./neural_net.h"

// Model * init(){
//   Model * model = (Model*)malloc(sizeof(Model));
//   model->next = NULL;
//   model->prev = NULL;
//   model->type = NONE;
//   model->DENSE = NULL;
//   return model;
// }

// Model * new_node(void * layer, char * type){
//   Model * model = (Model*)malloc(sizeof(Model));
//   model->next = model->prev = NULL;
//   if(!strcmp(type,"Dense")){
//     model->DENSE = (Dense_layer*)layer;
//     model->type = DENSE;
//   } 
//   return model;
// }

// void append_graph(Model * model, void * layer, char * type){
//   Model * new = new_node(layer,type);
//   model->next = new;
//   new->prev = model;
//   return;
// }

// void destroy_model(Model * model){
//   Model * temp = model;
//   Model * prev;
//   while(temp!=NULL){
//     prev = temp;
//     temp = temp->next;
//     free(prev->DENSE);
//     free(prev);
//   }
//   prev = NULL;
//   temp = NULL;
//   model = NULL;
//   return;
// }

// void printModel(Model * model){
//   Model * temp = model;
//   while(temp!=NULL){
//     if(model->type)
//       printf("%s\n",model->type);
//     printf("NULL\n");
//   }
// }

// int main(){
//   model = init();
//   Dense(.layer_size=30,.activation="relu",.initializer="he");
//   printModel(model);
// }