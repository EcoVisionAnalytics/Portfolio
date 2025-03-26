

if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(rvest)) install.packages("rvest", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(maps)) install.packages("maps", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(ggiraph)) install.packages("ggiraph", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(leaflet)) install.packages("leaflet", repos = "http://cran.us.r-project.org")
if(!require(plotly)) install.packages("plotly", repos = "http://cran.us.r-project.org")
if(!require(geojsonio)) install.packages("geojsonio", repos = "http://cran.us.r-project.org")
if(!require(shiny)) install.packages("shiny", repos = "http://cran.us.r-project.org")
if(!require(shinyWidgets)) install.packages("shinyWidgets", repos = "http://cran.us.r-project.org")
if(!require(shinydashboard)) install.packages("shinydashboard", repos = "http://cran.us.r-project.org")
if(!require(shinythemes)) install.packages("shinythemes", repos = "http://cran.us.r-project.org")
if(!require(eks)) install.packages("eks", repos = "http://cran.us.r-project.org")
if(!require(colorspace)) install.packages("colorspace", repos = "http://cran.us.r-project.org")
if(!require(patchwork)) install.packages("patchwork", repos = "http://cran.us.r-project.org")


packageurl <- "https://cran.r-project.org/src/contrib/Archive/bbsBayes/bbsBayes_2.5.3.tar.gz"
install.packages(packageurl, repos=NULL, type="source")
library(bbsBayes)


one_col = "#cc4c02"
two_col = "#662506"
three_col = "#045a8d"
four_col = "#4d004b"
five_col = "#016c59"
bbsdata <- load_bbs_data()
gye <- get_composite_regions(strata_type = "bbs_usgs")
gye$GYE <- ifelse(test$region %in% c("US-WY-10", "US-MT-10", "US-MT-17", "US-ID-10", "US-ID-9"),"GYE","Other")


trend = function(species_select) {
  jags_data <- prepare_jags_data(strat_data = strat_data,
                                 species_to_run = "Barn Swallow",
                                 model = "gamye",
                                 min_max_route_years = 2,
                                 heavy_tailed = TRUE)
}


total_obs = function(species_select) {
  plot_df = subset(bbsbcr, bbsbcr$SciName == species_select)
  plot_df = aggregate(plot_df['Count'], by=plot_df['Year'], sum)
  ggplot(plot_df, aes(x = Year, y = Count)) + geom_line(colour = two_col) + geom_point(size = 1, alpha = 0.8, colour = two_col) +
    ylab("Total Counts (BBS + IMBCR)") + xlab("Year") + theme_bw() + 
    geom_smooth() +
    theme(legend.title = element_blank(), legend.position = "", plot.title = element_text(size=10), 
          plot.margin = margin(5, 5, 5, 5))
}



# Define UI for application that draws a histogram
ui <- bootstrapPage(
    navbarPage(theme = shinytheme("flatly"), collapsible = TRUE,
               HTML('<a style="text-decoration:none;cursor:default;color:#FFFFFF;" class="active" href="#">Ecological Health of GYE Birds</a>'), id="nav",
               windowTitle = "GYE Birds Dashboard",
               
               tabPanel("Species Trends",
                        
                        sidebarLayout(
                          sidebarPanel(
                            
                          
                            span(tags$i(h6("Output is a kernel density estimation for the selected year. Animate to see over time.")), style="color:#045a8d"),
                            
                            pickerInput("species_select", "Species:",   
                                        choices = unique(bbsbcr$SciName), 
                                        selected = c("Select Species name"),
                                        multiple = FALSE),
                          
                            
                            "If you receive an error, this means not data exists for the selected year. Data are a concatenation and summarized product of the BBS and IMBCR data sets.
                            Please see project", tags$a("Github", target = "_blank", href = "https://github.com/pythonpandas303/GYE_Birds"), "repo for data sources.
                            Advance slider forward until data appears (few seconds of loading time). Observations over Time plots are all counts of that species 
                            in the GYE, by year. Thus, date slider is not relevant for these plots. The year 2020 is likely to be an outlier in most species due to the COVID19 pandemic and it's impact on sampling efforts.",
                            width = 4 ),
                         
                          
                          mainPanel(
                            
                              tabPanel("Spatial Distribution in GYE over Time", plotlyOutput("spatial_dist_kde")),
                              tabsetPanel(
                              tabPanel("Observations over Time", plotlyOutput("total_obs"))
                            )
                          )
                        )
               )))

# Define server logic required to draw a histogram
server <- function(input, output, session) {

        
    output$total_obs <- renderPlotly({
    total_obs(species_select=input$species_select)
    })
    
    output$spatial_dist_kde <- renderPlotly({
      spatial_dist_kde(species_select=input$species_select, year_select = input$year_select)
    })
    
    

}

shinyApp(ui, server)
