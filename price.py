# ==============================
# Estimation du prix de réparation
# ==============================

def estimate_price(class_name, confidence):
    """
    Estime le prix de réparation selon le type de dommage
    """
    
    if class_name == "Intacte":
        return {
            "statut": "✅ Véhicule en bon état",
            "prix_min": 0,
            "prix_max": 0,
            "recommandation": "Aucune réparation nécessaire"
        }
    
    elif class_name == "Endommagée":
        if confidence >= 0.90:
            return {
                "statut": "🔴 Dommage Sévère",
                "prix_min": 10000,
                "prix_max": 30000,
                "recommandation": "Réparation majeure nécessaire — Contacter l'assurance"
            }
        elif confidence >= 0.75:
            return {
                "statut": "🟠 Dommage Modéré",
                "prix_min": 3000,
                "prix_max": 10000,
                "recommandation": "Réparation moyenne — Devis recommandé"
            }
        else:
            return {
                "statut": "🟡 Dommage Léger",
                "prix_min": 500,
                "prix_max": 3000,
                "recommandation": "Petite réparation — Polish ou carrosserie légère"
            }


def display_price(class_name, confidence):
    result = estimate_price(class_name, confidence)
    print("\n" + "="*50)
    print("       ESTIMATION DU PRIX DE RÉPARATION")
    print("="*50)
    print(f"  Statut       : {result['statut']}")
    print(f"  Prix estimé  : {result['prix_min']} — {result['prix_max']} DH")
    print(f"  Conseil      : {result['recommandation']}")
    print("="*50 + "\n")
    return result


# ==============================
# Test
# ==============================
if __name__ == "__main__":
    # تجربة
    display_price("Endommagée", 0.875)
    display_price("Intacte", 0.95)
    display_price("Endommagée", 0.60)