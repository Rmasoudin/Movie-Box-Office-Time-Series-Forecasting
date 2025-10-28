from graphviz import Digraph

dot = Digraph(comment='Movie Database Schema')
dot.attr(rankdir='LR', size='8,5')
dot.node('MPAA', '''MPAA
- id (PK)
- rating''')

dot.node('Movie', '''Movie
- id (PK)
- title
- genre
- release_date
- running_time
- budget
- domestic
- international
- in_release
- mpaa_id (FK)''')

dot.node('WeeklyRevenue', '''WeeklyRevenue
- id (PK)
- movie_id (FK)
- week_number
- revenue''')

dot.edge('MPAA', 'Movie', label='1 -> *')
dot.edge('Movie', 'WeeklyRevenue', label='1 -> *')
dot.render('schema_diagram', format='png', cleanup=True)
print("Schema diagram saved as 'schema_diagram.png'")