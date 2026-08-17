target_sources(GluonIR PRIVATE
  Ops.cpp
  Traits.cpp
  Types.cpp
)

add_dependencies(GluonIR GluonMetaXTableGen)
