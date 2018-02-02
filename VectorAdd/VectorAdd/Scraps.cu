
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITEMS 5
struct stock {

	char symbol[5];
	int quantity;
	float price;
	struct stock *next;
};

struct stock *first;
struct stock *current;
struct stock *newStock;
struct stock *make_structure(void);
void fill_structure(struct stock *a, int c);
void show_structure(struct stock *a);


//int main()
//{
//	printf("Itemainyy\n");
//}

//int main()
//{
//	int x;
//	for (x = 0; x<ITEMS; x++)
//	{
//		if (x == 0)
//		{
//			first = make_structure();
//			current = first;
//		}
//		else
//		{
//			newStock = make_structure();
//			current->next = newStock;
//			current = newStock;
//		}
//		fill_structure(current, x + 1);
//	}
//
//	//current->next = null;
//
//	/* Display database */
//	printf("Investment Portfolio\n");
//	printf("SymboltSharestPricetValuen");
//	current = first;
//	while (current)
//	{
//		show_structure(current);
//		current = current->next;
//	}
//	return(0);
//}


struct stock *make_structure(void)
{
	struct stock *a;
	a = (struct stock *)malloc(sizeof(struct stock));
	if (a == NULL)
	{
		puts("Some kind of malloc() error");
		exit(1);
	}
	return(a);
}
void fill_structure(struct stock *a, int c)
{
	printf("Item #%d/%d:n", c, ITEMS);
	printf("Stock Symbol: ");
	scanf("%s", a->symbol);
	printf("Number of shares: ");
	scanf("%d", &a->quantity);
	printf("Share price: ");
	scanf("%f", &a->price);
}

void show_structure(struct stock *a)
{
	printf("%-6st%5dt%.2ft%.2fn",
		a->symbol,
		a->quantity,
		a->price,
		a->quantity*a->price);
}




