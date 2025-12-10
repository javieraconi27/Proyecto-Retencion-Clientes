# ==============================================================================
# PROYECTO FINAL: DASHBOARD DE GESTIÓN DE RETENCIÓN DE CLIENTES (NIVEL EJECUTIVO)
# TALLER DE INTELIGENCIA DE NEGOCIOS - UDD
# ==============================================================================

# 1. SETUP GLOBAL Y LIBRERÍAS
# ------------------------------------------------------------------------------
library(shiny)
library(shinydashboard)
library(tidyverse)
library(lubridate)
library(plotly)
library(DT)
library(rsample)
library(recipes)
library(mice)
library(yardstick)
library(pROC)
library(ggplot2)
library(cluster) 

# --- Carga de Datos ---
# Ajusta la ruta si es necesario
df_raw <- read_csv("synthetic_customers_churn.csv", show_col_types = FALSE)

# Preprocesamiento Inicial
prep_data <- df_raw %>%
  mutate(
    join_date = ymd_hms(join_date),
    last_activity_date = ymd_hms(last_activity_date),
    tenure_days = interval(join_date, last_activity_date) / ddays(1),
    churn_month = floor_date(last_activity_date, "month"),
    # Definimos niveles: El primero es "Activo", el segundo "Fugado" (Evento de interés)
    churned = factor(churned, levels = c(0, 1), labels = c("Activo", "Fugado")),
    is_premium = as.factor(is_premium)
  ) %>%
  select(-customer_id, -tenure_months) 

# 2. MODELADO SUPERVISADO (Regresión Logística con Ingeniería de Variables)
# ------------------------------------------------------------------------------
set.seed(42)
split_data <- initial_split(prep_data, prop = 0.8, strata = churned)
train_raw <- training(split_data)
test_raw <- testing(split_data)

# Recipe con Normalización (Crucial para que el modelo funcione bien)
final_recipe <- recipe(churned ~ ., data = train_raw) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

prep_recipe <- prep(final_recipe)
train_processed <- bake(prep_recipe, new_data = train_raw)
test_processed <- bake(prep_recipe, new_data = test_raw)

# --- MODELO CON INTERACCIONES ---
# Agregamos interacciones clave para mejorar la capacidad de detección
model_glm <- glm(churned ~ . + 
                   support_tickets_12m * satisfaction_score + 
                   monthly_spend_usd * tenure_days, 
                 data = train_processed, family = "binomial")

# Cálculo de métricas
preds_test <- test_processed %>%
  mutate(
    prob_fuga = predict(model_glm, newdata = test_processed, type = "response")
  )

# --- CORRECCIÓN DE AUC ---
# Usamos event_level = "second" porque "Fugado" es el segundo nivel del factor
auc_val <- roc_auc(preds_test, truth = churned, prob_fuga, event_level = "second")$.estimate

# 3. DATOS PARA VISUALIZACIONES (KPIs y EDA)
# ------------------------------------------------------------------------------
# Recipe auxiliar SIN normalizar para mostrar valores reales ($) en los gráficos
recipe_display <- recipe(churned ~ ., data = train_raw) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

prep_display <- prep(recipe_display)
df_display <- bake(prep_display, new_data = prep_data) %>%
  mutate(
    monthly_spend_usd = as.numeric(monthly_spend_usd),
    satisfaction_score = as.numeric(satisfaction_score),
    GMAS = monthly_spend_usd * (satisfaction_score / 5.0)
  )

# Datos Tendencia
tcm_data <- prep_data %>%
  group_by(churn_month) %>%
  summarise(
    Total = n(),
    Fugados = sum(churned == "Fugado"),
    Tasa_Fuga = (Fugados / Total) * 100
  ) %>%
  filter(Total > 10, churn_month < max(prep_data$churn_month))

tasa_actual <- round(tail(tcm_data$Tasa_Fuga, 1), 2)
gasto_promedio <- round(mean(df_display$monthly_spend_usd[df_display$churned == "Activo"]), 1)

# 4. INTERFAZ DE USUARIO (UI)
# ------------------------------------------------------------------------------

ui <- dashboardPage(
  skin = "black", 
  dashboardHeader(title = "Gestión de Clientes", titleWidth = 250),
  
  dashboardSidebar(
    width = 250,
    sidebarMenu(
      menuItem("Inicio (Contexto)", tabName = "home", icon = icon("home")),
      menuItem("Análisis Exploratorio", tabName = "eda", icon = icon("search")),
      menuItem("KPIs de Negocio", tabName = "kpis", icon = icon("chart-line")),
      menuItem("Segmentación Clientes", tabName = "clusters", icon = icon("users")),
      menuItem("Simulador de Riesgo", tabName = "predict", icon = icon("calculator"))
    ),
    hr(),
    div(style = "padding: 15px",
        h5("Filtros Globales", style="color:white;"),
        selectInput("region_filter", "Región", choices = c("Todas", unique(as.character(df_display$region))), selected = "Todas"),
        selectInput("segment_filter", "Segmento", choices = c("Todos", unique(as.character(df_display$segment))), selected = "Todos"),
        dateRangeInput("date_filter", "Periodo", start = min(prep_data$last_activity_date), end = max(prep_data$last_activity_date))
    )
  ),
  
  dashboardBody(
    tabItems(
      
      # --- TAB 1: HOME ---
      tabItem(tabName = "home",
              h2("Contexto Estratégico del Proyecto"),
              fluidRow(
                box(title = "Diagnóstico y Problemática de Negocio", width = 12, status = "danger", solidHeader = TRUE,
                    h4(icon("exclamation-triangle"), "El Desafío: Erosión de Ingresos Recurrentes"),
                    p("La organización enfrenta una pérdida sostenida de clientes (Churn), lo que impacta directamente en el margen de utilidad y aumenta la presión sobre el equipo de ventas para captar nuevos usuarios."),
                    p("Actualmente, la gestión de retención es **reactiva**: se intenta salvar al cliente cuando ya ha decidido irse. Además, la falta de segmentación impide diferenciar entre la pérdida de un cliente de 'Alto Valor' versus uno de 'Bajo Margen'."),
                    tags$ul(
                      tags$li(tags$b("Impacto Financiero:"), "Adquirir un nuevo cliente cuesta entre 5 y 7 veces más que retener a uno existente."),
                      tags$li(tags$b("Punto Ciego:"), "No existen alertas tempranas basadas en datos para los equipos de Customer Success.")
                    )
                )
              ),
              fluidRow(
                box(title = "Objetivo de la Solución Analítica", width = 12, status = "success", solidHeader = TRUE,
                    h4(icon("bullseye"), "Estrategia: De la Reacción a la Predicción"),
                    p("Este dashboard integra modelos de Machine Learning Supervisados y No Supervisados para transformar los datos históricos en decisiones accionables."),
                    tags$ul(
                      tags$li(tags$b("Visibilidad (KPIs):"), "Monitoreo en tiempo real de la tasa de fuga y satisfacción por región."),
                      tags$li(tags$b("Inteligencia (Clustering):"), "Identificación automática de perfiles de clientes para personalizar la oferta comercial."),
                      tags$li(tags$b("Acción (Predicción):"), "Evaluación del riesgo de fuga individual (Scoring) para priorizar recursos en clientes críticos.")
                    )
                )
              ),
              fluidRow(
                box(title = "Diccionario de Variables Clave", width = 12, status = "info", collapsible = TRUE, collapsed = FALSE,
                    tableOutput("dict_table")
                )
              )
      ),
      
      # --- TAB 2: EDA ---
      tabItem(tabName = "eda",
              h2("Análisis del Comportamiento"),
              fluidRow(
                box(title = "Fuga por Campaña", width = 6, status = "primary", plotlyOutput("plot_campaign", height = 300)),
                box(title = "Satisfacción por Segmento", width = 6, status = "primary", plotlyOutput("plot_satisfaction", height = 300))
              ),
              fluidRow(
                box(title = "Mapa de Riesgo (Facturación vs. Antigüedad)", width = 9, status = "warning", plotlyOutput("plot_scatter", height = 400)),
                box(title = "Insights", width = 3, background = "yellow",
                    h4("Conclusiones:"),
                    p("1. Fuga concentrada en baja antigüedad (< 1 año)."),
                    p("2. Alerta en clientes de alta facturación que se van rápido.")
                )
              )
      ),
      
      # --- TAB 3: KPIs ---
      tabItem(tabName = "kpis",
              h2("Tablero de Control Estratégico"),
              
              # Fila 1: Indicadores Críticos
              fluidRow(
                valueBoxOutput("kpi_tasa_fuga", width = 4),
                valueBoxOutput("kpi_num_fuga", width = 4), 
                valueBoxOutput("kpi_precision_modelo", width = 4)
              ),
              
              # Fila 2: Indicadores de Volumen y Valor
              fluidRow(
                valueBoxOutput("kpi_total_clientes", width = 4), 
                valueBoxOutput("kpi_gasto_promedio", width = 4),
                valueBoxOutput("kpi_gmas", width = 4) 
              ),
              
              fluidRow(
                box(title = "Tendencia de Fuga Histórica", width = 12, status = "danger", solidHeader = TRUE,
                    plotlyOutput("plot_trend", height = 350),
                    p("Línea roja: Umbral máximo tolerado (2%).")
                )
              ),
              fluidRow(
                box(title = "KPIs por Región", width = 12, DTOutput("table_kpis_region"))
              )
      ),
      
      # --- TAB 4: CLUSTERING ---
      tabItem(tabName = "clusters",
              h2("Segmentación Automática (Clustering)"),
              fluidRow(
                box(width = 12, p("Agrupación de clientes basada en comportamiento (Gasto, Satisfacción, Antigüedad) mediante IA."))
              ),
              fluidRow(
                box(title = "Mapa de Segmentos", width = 8, status = "success", plotlyOutput("plot_clusters", height = 450)),
                box(title = "Perfiles", width = 4, status = "success",
                    h4("Grupos Identificados:"),
                    p(tags$b("Grupo 1 (Rojo) - Vulnerables:"), "Bajo gasto, satisfacción media-baja."),
                    p(tags$b("Grupo 2 (Verde) - Consolidados:"), "Alta facturación y satisfacción. Clientes VIP."),
                    p(tags$b("Grupo 3 (Azul) - En Desarrollo:"), "Gasto medio, antigüedad creciente.")
                )
              ),
              fluidRow(
                box(title = "Estadísticas por Perfil", width = 12, DTOutput("table_cluster_stats"))
              )
      ),
      
      # --- TAB 5: SIMULADOR ---
      tabItem(tabName = "predict",
              h2("Simulador de Riesgo Individual"),
              fluidRow(
                box(title = "Perfil del Cliente / Prospecto", width = 4, status = "primary", solidHeader = TRUE,
                    p(tags$small(icon("info-circle"), " Configuración por defecto: PROSPECTO (Cliente Nuevo).")),
                    br(),
                    numericInput("p_spend", "Facturación Mensual Estimada ($)", value = 50, min=0),
                    selectInput("p_region", "Región", choices = unique(df_display$region)),
                    selectInput("p_segment", "Segmento", choices = unique(df_display$segment)),
                    hr(),
                    h4("Comportamiento (Estado Cero)"),
                    numericInput("p_tenure", "Antigüedad (Días)", value = 1, min=0),
                    numericInput("p_tickets", "Reclamos Históricos", value = 0, min=0),
                    numericInput("p_satisfaction", "Satisfacción (1-5)", value = 3.0, step=0.1, min=1, max=5),
                    
                    box(title = "Opciones Avanzadas", width = 12, collapsible = TRUE, collapsed = TRUE, solidHeader = FALSE, background = "light-blue",
                        numericInput("p_age", "Edad", value = 30),
                        numericInput("p_products", "Productos", value = 1),
                        selectInput("p_campaign", "Última Campaña", choices = unique(as.character(df_display$last_campaign)), selected = "Ninguna"),
                        selectInput("p_premium", "¿Es Premium?", choices = unique(as.character(df_display$is_premium))),
                        numericInput("p_credit", "Score Crédito", value = 650)
                    ),
                    br(),
                    actionButton("calc_btn", "CALCULAR RIESGO", class = "btn-warning btn-lg btn-block", icon = icon("calculator"))
                ),
                
                box(title = "Diagnóstico de Retención", width = 8, status = "warning", solidHeader = TRUE,
                    fluidRow(
                      valueBoxOutput("box_risk_result", width = 6),
                      valueBoxOutput("box_risk_label", width = 6)
                    ),
                    fluidRow(
                      column(12, align="center",
                             h4("Termómetro de Fuga"),
                             plotlyOutput("plot_gauge", height = "250px")
                      )
                    ),
                    hr(),
                    div(class = "alert alert-info", role = "alert",
                        h4(icon("clipboard-check"), "Estrategia Recomendada:"),
                        textOutput("recommendation_text")
                    )
                )
              )
      )
    )
  )
)

# 5. SERVER
# ------------------------------------------------------------------------------

server <- function(input, output) {
  
  # --- Diccionario ---
  output$dict_table <- renderTable({
    data.frame(
      Variable = c("churned", "monthly_spend_usd", "tenure_days", "satisfaction_score", "support_tickets_12m"),
      Descripción = c("Estado (Activo/Fugado).", "Facturación Mensual ($).", "Días de antigüedad.", "Satisfacción (1-5).", "Número de reclamos en el último año.")
    )
  }, striped = TRUE)
  
  # --- Filtro Reactivo ---
  data_filtered <- reactive({
    d <- df_display
    if(input$region_filter != "Todas") d <- d %>% filter(region == input$region_filter)
    if(input$segment_filter != "Todos") d <- d %>% filter(segment == input$segment_filter)
    d <- d %>% filter(last_activity_date >= input$date_filter[1] & last_activity_date <= input$date_filter[2])
    return(d)
  })
  
  # --- EDA ---
  output$plot_campaign <- renderPlotly({
    d <- data_filtered() %>% count(last_campaign, churned) %>% group_by(last_campaign) %>% mutate(pct = n/sum(n))
    ggplotly(ggplot(d, aes(x = last_campaign, y = n, fill = churned)) + geom_col(position = "fill") + scale_fill_manual(values = c("Activo"="#00a65a", "Fugado"="#dd4b39")) + labs(y="Proporción", x="Campaña") + theme_minimal())
  })
  
  output$plot_satisfaction <- renderPlotly({
    ggplotly(ggplot(data_filtered(), aes(x = segment, y = satisfaction_score, fill = segment)) + geom_boxplot() + theme_minimal())
  })
  
  output$plot_scatter <- renderPlotly({
    d_plot <- data_filtered() %>% sample_n(min(500, nrow(.)))
    ggplotly(ggplot(d_plot, aes(x = tenure_days, y = monthly_spend_usd, color = churned)) + geom_point(alpha=0.6) + scale_color_manual(values = c("Activo"="#00a65a", "Fugado"="#dd4b39")) + labs(x="Antigüedad", y="Facturación") + theme_minimal())
  })
  
  # --- KPIs ---
  
  # 1. Tasa de Fuga
  output$kpi_tasa_fuga <- renderValueBox({
    val <- round(mean(data_filtered()$churned == "Fugado") * 100, 2)
    color <- if(val > 2.0) "red" else "green"
    valueBox(paste0(val, "%"), "Tasa de Fuga (Periodo)", icon = icon("user-times"), color = color)
  })
  
  # 2. N° Fugas (Absoluto)
  output$kpi_num_fuga <- renderValueBox({
    val <- sum(data_filtered()$churned == "Fugado")
    valueBox(val, "Clientes Perdidos (N°)", icon = icon("user-minus"), color = "red")
  })
  
  # 3. Precisión Modelo
  output$kpi_precision_modelo <- renderValueBox({
    valueBox(paste0(round(auc_val, 2)*100, "%"), "Precisión Predictiva (AUC)", icon = icon("crosshairs"), color = "purple")
  })
  
  # 4. Total Clientes
  output$kpi_total_clientes <- renderValueBox({
    val <- nrow(data_filtered())
    valueBox(val, "Cartera Total (Selección)", icon = icon("users"), color = "aqua")
  })
  
  # 5. Ticket Promedio
  output$kpi_gasto_promedio <- renderValueBox({
    val <- round(mean(data_filtered()$monthly_spend_usd, na.rm=T), 1)
    valueBox(paste0("$", val), "Ticket Promedio", icon = icon("dollar-sign"), color = "blue")
  })
  
  # 6. GMAS
  output$kpi_gmas <- renderValueBox({
    val <- round(mean(data_filtered()$GMAS, na.rm=T), 1)
    color <- if(val > 60) "green" else "yellow"
    valueBox(val, "Valor GMAS (Calidad)", icon = icon("star"), color = color)
  })
  
  output$plot_trend <- renderPlotly({
    ggplotly(ggplot(tcm_data, aes(x = churn_month, y = Tasa_Fuga)) + geom_line(color="#dd4b39", size=1) + geom_hline(yintercept=2, linetype="dashed") + theme_minimal())
  })
  
  output$table_kpis_region <- renderDT({
    data_filtered() %>% group_by(region) %>% summarise(Clientes=n(), Fuga_Pct=round(mean(churned=="Fugado")*100,1), Gasto=round(mean(monthly_spend_usd),1)) %>% datatable(options=list(dom='t'), rownames=FALSE)
  })
  
  # --- CLUSTERING ---
  clusters_res <- reactive({
    d <- data_filtered() %>% select(monthly_spend_usd, satisfaction_score, tenure_days) %>% na.omit()
    d_scaled <- scale(d)
    set.seed(123)
    km <- kmeans(d_scaled, centers = 3)
    d$Cluster <- as.factor(km$cluster)
    d
  })
  output$plot_clusters <- renderPlotly({
    d <- clusters_res()
    ggplotly(ggplot(d, aes(x=satisfaction_score, y=monthly_spend_usd, color=Cluster)) + geom_point(alpha=0.7) + scale_color_manual(values=c("1"="#dd4b39","2"="#00a65a","3"="#3c8dbc")) + theme_minimal())
  })
  output$table_cluster_stats <- renderDT({
    clusters_res() %>% group_by(Cluster) %>% summarise(N=n(), Gasto=round(mean(monthly_spend_usd),1), Satisfaccion=round(mean(satisfaction_score),1), Antiguedad=round(mean(tenure_days),0)) %>% datatable(options=list(dom='t'), rownames=FALSE)
  })
  
  # --- SIMULADOR ---
  prob_reactiva <- eventReactive(input$calc_btn, {
    # 1. Input Data
    input_data <- data.frame(
      age = as.numeric(input$p_age),
      region = factor(input$p_region, levels = levels(prep_data$region)),
      segment = factor(input$p_segment, levels = levels(prep_data$segment)),
      monthly_spend_usd = as.numeric(input$p_spend),
      products_owned = as.numeric(input$p_products),
      support_tickets_12m = as.numeric(input$p_tickets),
      last_campaign = factor(input$p_campaign, levels = levels(prep_data$last_campaign)),
      satisfaction_score = as.numeric(input$p_satisfaction),
      credit_score = as.numeric(input$p_credit),
      is_premium = factor(input$p_premium, levels = levels(prep_data$is_premium)),
      tenure_days = as.numeric(input$p_tenure),
      churned = factor("Activo", levels = levels(prep_data$churned)), # Dummy
      join_date = Sys.time(), last_activity_date = Sys.time(), churn_month = floor_date(Sys.time(), "month")
    )
    # 2. BAKE (Normalizar)
    input_processed <- bake(prep_recipe, new_data = input_data)
    # 3. Predict (usando modelo con interacciones)
    prob <- predict(model_glm, newdata = input_processed, type = "response")
    return(as.numeric(prob))
  }, ignoreNULL=FALSE)
  
  output$box_risk_result <- renderValueBox({
    pct <- round(prob_reactiva() * 100, 1)
    color <- if(pct < 30) "green" else if (pct < 60) "yellow" else "red"
    valueBox(paste0(pct, "%"), "Probabilidad Calculada", icon = icon("percent"), color = color)
  })
  
  output$box_risk_label <- renderValueBox({
    pct <- round(prob_reactiva() * 100, 1)
    if(pct < 30) { txt <- "BAJO RIESGO"; col <- "green"; ico <- icon("smile") }
    else if (pct < 60) { txt <- "RIESGO MEDIO"; col <- "yellow"; ico <- icon("meh") }
    else { txt <- "ALTO RIESGO"; col <- "red"; ico <- icon("frown") }
    valueBox(txt, "Clasificación", icon = ico, color = col)
  })
  
  output$plot_gauge <- renderPlotly({
    val <- prob_reactiva() * 100
    plot_ly(
      domain = list(x = c(0, 1), y = c(0, 1)),
      value = val, title = list(text = "Nivel de Riesgo"),
      type = "indicator", mode = "gauge+number",
      gauge = list(
        axis = list(range = list(NULL, 100)), bar = list(color = "black"),
        steps = list(list(range = c(0, 30), color = "#00a65a"), list(range = c(30, 60), color = "#f39c12"), list(range = c(60, 100), color = "#dd4b39")),
        threshold = list(line = list(color = "red", width = 4), thickness = 0.75, value = val)
      )
    ) %>% layout(margin = list(t=20, b=20))
  })
  
  output$recommendation_text <- renderText({
    val <- prob_reactiva() * 100
    if(val > 60) "⚠️ ALERTA CRÍTICA: Cliente crítico. Contactar urgente para retención (Descuento/Upgrade)."
    else if (val > 30) "⚠️ PRECAUCIÓN: En zona de riesgo. Incluir en campaña de fidelización."
    else "✅ SEGURO: Cliente estable. Ideal para Venta Cruzada."
  })
}

shinyApp(ui, server)